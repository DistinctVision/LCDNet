import argparse
import os
from typing import Union, Optional, List, Tuple, Dict
from pathlib import Path
import time
from collections import OrderedDict

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

import plotly.graph_objects as go


def main(dataset: AstralDatasetReader, weights_path: Union[Path, str]):

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

    train_positions: List[v3d.Vector3d] = []
    train_transforms: List[np.ndarray] = []
    train_embeddings: List[np.ndarray] = []
    train_pc_data: List[Tuple[o3d.geometry.PointCloud, reg_module.Feature]] = []

    prev_position: Optional[v3d.Vector3d] = None

    number_of_frames = min(3000, len(dataset))

    with torch.no_grad():
        for i in tqdm(range(number_of_frames), 'train'):
            if i % 2 == 1:
                continue
            data = dataset[i]

            frame_transform = dataset.transform_manager.get_transform('ld_cc', 'map')
            position = v3d.create(frame_transform[0, 3], frame_transform[1, 3], frame_transform[2, 3])

            if prev_position is not None:
                delta_distance = v3d.length(v3d.sub(position, prev_position))
                if delta_distance < 2.0:
                    continue

            image = data['fc_near']
            query_pcd = torch.from_numpy(data['ld_cc']).cuda()
            model_input = model.backbone.prepare_input(query_pcd)

            model_input = KittiDataset.collate_batch([model_input])
            for key, val in model_input.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_input[key] = torch.from_numpy(val).float().cuda()
            batch_dict = model(model_input, metric_head=False, compute_rotation=False, compute_transl=False)
            embedding = batch_dict['out_embedding'][0].cpu().numpy()

            coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
            features = batch_dict['point_features_NV'].squeeze(-1)
            coords = coords[0]
            features = features[0]

            query_pcd = o3d.geometry.PointCloud()
            query_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy())
            query_pcd_feature = reg_module.Feature()
            query_pcd_feature.data = features.permute(0, 1).cpu().numpy()

            train_positions.append(position)
            train_transforms.append(frame_transform)
            train_embeddings.append(embedding)
            train_pc_data.append((query_pcd, query_pcd_feature))

            # cv2.imshow('image', image)
            # cv2.waitKey(13)

            prev_position = position

    prev_position = None

    print('Query step..')

    # quantiser = faiss.IndexFlatL2(256)
    # index = faiss.IndexIVFFlat(quantiser, 256, 100)
    # index.train(np.array(train_embeddings))
    index = faiss.IndexFlatL2(256)
    index.add(np.array(train_embeddings))

    output: Dict[str, List[float]] = {
            'distance': [],
            'f_delta': [],
            'fitness': [],
            'inlier_rmse': [],
            'refined_distance': []
    }
    with torch.no_grad():
        for i in tqdm(range(number_of_frames), 'query'):
            if i % 2 == 0:
                continue
            data = dataset[i]

            frame_transform = dataset.transform_manager.get_transform('ld_cc', 'map')
            position = v3d.create(frame_transform[0, 3], frame_transform[1, 3], frame_transform[2, 3])

            if prev_position is not None:
                delta_distance = v3d.length(v3d.sub(position, prev_position))
                if delta_distance < 2.0:
                    continue

            image = data['fc_near']
            query_pcd = torch.from_numpy(data['ld_cc']).cuda()
            model_input = model.backbone.prepare_input(query_pcd)

            model_input = KittiDataset.collate_batch([model_input])
            for key, val in model_input.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_input[key] = torch.from_numpy(val).float().cuda()
            batch_dict = model(model_input, metric_head=False, compute_rotation=False, compute_transl=False)
            embedding = batch_dict['out_embedding'][0].cpu().numpy()

            coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
            features = batch_dict['point_features_NV'].squeeze(-1)
            coords = coords[0]
            features = features[0]

            query_pcd = o3d.geometry.PointCloud()
            query_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy())
            query_pcd_feature = reg_module.Feature()
            query_pcd_feature.data = features.permute(0, 1).cpu().numpy()

            d, i = index.search(np.array([embedding]), 2)
            d, i = d[0], i[0]
            distances = []
            for idx in i:
                train_position = train_positions[idx]
                distance = v3d.length(v3d.sub(train_position, position))
                distances.append(distance)
            min_dis, max_dis = min(distances), max(distances)

            train_frame_transform = train_transforms[i[0]]
            train_pcd, train_pcd_feature = train_pc_data[i[0]]

            result: o3d.pipelines.registration.RegistrationResult = \
                reg_module.registration_ransac_based_on_feature_matching(
                query_pcd, train_pcd, query_pcd_feature, train_pcd_feature, True,
                0.6,
                reg_module.TransformationEstimationPointToPoint(False),
                3, [],
                reg_module.RANSACConvergenceCriteria(5000))

            if result.fitness < 0.8 or result.inlier_rmse > 0.2:
                continue

            transformation_1 = np.dot(train_frame_transform, result.transformation)

            position_1 = v3d.create(transformation_1[0, 3], transformation_1[1, 3], transformation_1[2, 3])

            d1 = v3d.length(v3d.sub(position_1, position))

            output['distance'].append(distances[0])
            output['f_delta'].append(max_dis - min_dis)
            output['fitness'].append(result.fitness)
            output['inlier_rmse'].append(result.inlier_rmse)
            output['refined_distance'].append(d1)

            # cv2.imshow('image', image)
            # cv2.waitKey(13)

    fig = go.Figure()
    size = len(output['distance'])
    for out_name, out_list in output.items():
        fig.add_trace(go.Scatter(x=np.arange(size), y=out_list, mode='lines', name=out_name))
    fig.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='', help="Path to dataset")
    parser.add_argument('--dataset_id', default='', help='Dataset id')
    parser.add_argument('--weights_path', default='', required=True, help='Path to model weights')
    args = parser.parse_args()

    if args.dataset_id:
        clearml_dataset = clearml.Dataset.get(dataset_id=args.dataset_id)
        dataset_path = Path(clearml_dataset.get_local_copy())
    elif args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        raise 'Dataset is not set - "dataset_path" or "dataset_id" argument is needed'

    dataset = AstralDatasetReader(dataset_path, 'm11', ['ld_cc', 'fc_near'])

    main(dataset, args.weights_path)
