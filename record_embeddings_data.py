import argparse
import os
from typing import Union, Optional, List, Tuple, Dict
from pathlib import Path
import time
from collections import OrderedDict
from copy import copy
from attrs import define

import faiss
import open3d as o3d
if hasattr(o3d, 'pipelines'):
    reg_module = o3d.pipelines.registration
else:
    reg_module = o3d.registration
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import random

import clearml

from tqdm import tqdm
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from datasets.astral_dataset_reader import AstralDatasetReader

from models.get_models import get_model

import cv2
from mldatatools.utils import v3d
from mldatatools.utils.visualizer3d import Visualizer3d

import plotly.graph_objects as go


import rospy
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import starline_msgs.srv

from utils.ros_utils import array_to_pointcloud2, pointcloud2_to_xyz_array


@define
class FrameRecord:
    transform: v3d.Vector3d
    embedding: np.ndarray
    point_cloud: o3d.geometry.PointCloud
    point_features: reg_module.Feature


def record_embeddings(dataset: AstralDatasetReader, weights_path: Union[Path, str], visualize: bool):
    try:
        rospy.wait_for_service('/lidar_apollo_instance_segmentation/dynamic_objects', timeout=rospy.Duration(secs=3))
    except Exception:
        print('Not found service: /lidar_apollo_instance_segmentation/dynamic_objects')
        return
    lidar_segmentation_service = rospy.ServiceProxy('/lidar_apollo_instance_segmentation/dynamic_objects',
                                                    starline_msgs.srv.Cloud2DynamicObjects)

    saved_params = torch.load(weights_path, map_location='cpu')

    exp_cfg = saved_params['config']
    exp_cfg['batch_size'] = 1
    exp_cfg['loop_file'] = 'loop_GT_4m'
    exp_cfg['head'] = 'UOTHead'
    exp_cfg['PC_RANGE'] = [-70.4, -70.4, -1, 70.4, 70.4, 3]

    model = get_model(exp_cfg, is_training=False)
    renamed_dict = OrderedDict()
    for key in saved_params['state_dict']:
        if not key.startswith('module'):
            renamed_dict = saved_params['state_dict']
            break
        else:
            renamed_dict[key[7:]] = saved_params['state_dict'][key]

    # Convert shape from old OpenPCDet
    if renamed_dict['backbone.backbone.conv_input.0.weight'].shape != \
            model.state_dict()['backbone.backbone.conv_input.0.weight'].shape:
        for key in renamed_dict:
            if key.startswith('backbone.backbone.conv') and key.endswith('weight'):
                if len(renamed_dict[key].shape) == 5:
                    renamed_dict[key] = renamed_dict[key].permute(-1, 0, 1, 2, 3)

    res = model.load_state_dict(renamed_dict, strict=True)

    if len(res[0]) > 0:
        print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

    model = model.to('cuda:0')
    model.eval()

    prev_position: Optional[v3d.Vector3d] = None

    ui_pcd_id: Optional[str] = None
    visualizer = None

    key_distance_step = 10.0

    semantic_colors = [(1, 1, 1),  # UNKNOWN
                       (0, 1, 0),  # CAR
                       (0.5, 1, 0),  # TRUCK
                       (0, 1, 0.5),  # BUS
                       (0, 0.5, 1),  # BICYCLE
                       (1, 0.5, 1),  # MOTORBIKE
                       (1, 0, 0),  # PEDESTRIAN
                       (1, 0.5, 0)  # ANIMAL
                       ]

    if visualize:
        visualizer = Visualizer3d()
        visualizer.start()

    ui_dynamic_objects = []

    frame_records: List[FrameRecord] = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            dataset.update_transform(i)

            frame_transform = dataset.transform_manager.get_transform('ld_cc', 'map')
            position = v3d.create(*frame_transform[:3, 3])

            if prev_position is not None:
                key_frame_distance = v3d.length(v3d.sub(prev_position, position))
                if key_frame_distance < key_distance_step:
                    continue

            data = dataset[i]

            pcd = data['ld_cc']
            f_cloud_msg = array_to_pointcloud2(pcd.astype(np.float32), ['x', 'y', 'z', 'intensity'], frame_id='ld_cc')
            segmentation = lidar_segmentation_service(f_cloud_msg)
            for f_obj in segmentation.objects.feature_objects:
                aabb = f_obj.object.shape.aabb
                pcd = pcd[np.logical_or(np.logical_or(pcd[:, 0] < aabb.min_x, pcd[:, 0] > aabb.max_x),
                                            np.logical_or(pcd[:, 1] < aabb.min_y, pcd[:, 1] > aabb.max_y))]

            query_pcd = torch.from_numpy(pcd).cuda()
            model_input = model.backbone.prepare_input(query_pcd)

            model_input = KittiDataset.collate_batch([model_input])
            for key, val in model_input.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_input[key] = torch.from_numpy(val).float().cuda()
            batch_dict = model(model_input, metric_head=False, compute_rotation=False, compute_transl=False)
            embedding = batch_dict['out_embedding'][0].cpu().numpy()

            points = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
            point_features = batch_dict['point_features_NV'].squeeze(-1)
            points = points[0]
            point_features = point_features[0]
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points[:, 1:].cpu().numpy())

            frame_record = FrameRecord(frame_transform, embedding, point_cloud, point_features)
            frame_records.append(frame_record)

            prev_position = copy(position)
            if visualizer:
                for ui_dynamic_obj in ui_dynamic_objects:
                    visualizer.remove_geometry(ui_dynamic_obj)
                if ui_pcd_id:
                    visualizer.remove_geometry(ui_pcd_id)
                ui_pcd_id = visualizer.add_geometry(point_cloud)
                ui_dynamic_objects = []
                for f_obj in segmentation.objects.feature_objects:
                    aabb = f_obj.object.shape.aabb
                    cube = o3d.geometry.TriangleMesh.create_box((aabb.max_x - aabb.min_x), (aabb.max_y - aabb.min_y))
                    cube.translate((aabb.min_x, aabb.min_y, 1.0))
                    cube.paint_uniform_color(semantic_colors[f_obj.object.semantic.type])
                    ui_dynamic_objects.append(visualizer.add_geometry(cube, 'obj'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='', help="Path to dataset")
    parser.add_argument('--dataset_id', default='', help='Dataset id')
    parser.add_argument('--weights_path', default='checkpoints/LCDNet-kitti360.tar', help='Path to model weights')
    parser.add_argument('--location', type=str, default='m11', help='A location name for GNSS')
    args = parser.parse_args()

    if args.dataset_id:
        clearml_dataset = clearml.Dataset.get(dataset_id=args.dataset_id)
        dataset_path = Path(clearml_dataset.get_local_copy())
    elif args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        raise 'Dataset is not set - "dataset_path" or "dataset_id" argument is needed'

    dataset = AstralDatasetReader(dataset_path, args.location, ['ld_cc', 'fc_near'])

    record_embeddings(dataset, args.weights_path, visualize=True)
