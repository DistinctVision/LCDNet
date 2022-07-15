import sys
from typing import Union, TypeVar, Tuple, Optional
from pathlib import Path
from collections import OrderedDict

import clearml
import torch

import numpy as np
import open3d as o3d
reg_module = o3d.pipelines.registration
from models.get_models import get_model
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from mldatatools.dataset import Dataset
from mldatatools.dataset import Frame, Message, Sensor, NumpyAnnotation, NumpyFileAnnotation, GlobalPoseMsgData

import hnswlib

import rospy
import starline_msgs.srv
from utils.ros_utils import array_to_pointcloud2


MessageT = TypeVar('MessageT', bound=Message)
SensorT = TypeVar('SensorT', bound=Sensor)


class LcdDbDataset:
    def __init__(self,
                 dataset_id: Union[Path, str],
                 location: str):
        super().__init__()

        clearml_dataset = clearml.Dataset.get(dataset_id=dataset_id)
        dataset_path = Path(clearml_dataset.get_local_copy())

        self.model = self._load_lcd_net()

        self.location = location
        self.astral_dataset = Dataset.load(dataset_path, check=False)

        self.lidar = self._get_sensor_by_name('ld_cc')
        self.gnss = self._get_sensor_by_name('nmea')

        self.embeddings = [self._get_embedding_from_frame(frame).array for frame in self.astral_dataset.iter_frames()]
        self.geo_poses = [self._get_msg_by_sensor(frame, self.gnss).data for frame in self.astral_dataset.iter_frames()]

        indices = [i for i in range(len(self.embeddings))]

        self.index = hnswlib.Index(space='l2', dim=256)
        self.index.init_index(len(self.embeddings))
        self.index.add_items(self.embeddings, indices)

        rospy.wait_for_service('/lidar_apollo_instance_segmentation/dynamic_objects', timeout=rospy.Duration(secs=3))
        self.lidar_segmentation_service = rospy.ServiceProxy('/lidar_apollo_instance_segmentation/dynamic_objects',
                                                             starline_msgs.srv.Cloud2DynamicObjects)

        # self.visualizer = Visualizer3d()
        # self.visualizer.start()

    def _load_lcd_net(self, weights_path: Union[Path, str] = 'checkpoints/LCDNet-kitti360.tar'):
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

        return model

    @staticmethod
    def _get_embedding_from_frame(frame: Frame) -> Optional[NumpyAnnotation]:
        for pkg in frame.packages:
            if pkg.class_name == 'embedding':
                return pkg.data
        return None

    def _get_sensor_by_name(self, sensor_name: str) -> SensorT:
        return self.astral_dataset.annotation.sensors[
            [x.name for x in self.astral_dataset.annotation.sensors].index(sensor_name)]

    @staticmethod
    def _get_msg_by_sensor(frame: Frame, sensor: Sensor) -> MessageT:
        return frame.msg_ids[[msg.value.sensor_id for msg in frame.msg_ids].index(sensor.id)].value

    def _get_data_for_frame(self, frame_index: int) -> Tuple[o3d.geometry.PointCloud, reg_module.Feature]:
        frame = self.astral_dataset.get_frame(frame_index)
        point_cloud_msg = self._get_msg_by_sensor(frame, self.lidar)
        point_cloud = point_cloud_msg.data.load(self.astral_dataset.root_dir, remove_nan_points=True)
        point_features: Optional[reg_module.Feature] = None
        for pkg in frame.packages:
            if pkg.class_name == 'pcd_features':
                point_features = pkg.data.load(self.astral_dataset.root_dir)
        return point_cloud, point_features

    def __call__(self, pcd: np.ndarray) -> Tuple[Optional[GlobalPoseMsgData], Optional[np.ndarray]]:

        f_cloud_msg = array_to_pointcloud2(pcd.astype(np.float32), ['x', 'y', 'z', 'intensity'], frame_id='ld_cc')
        segmentation = self.lidar_segmentation_service(f_cloud_msg)
        for f_obj in segmentation.objects.feature_objects:
            aabb = f_obj.object.shape.aabb
            pcd = pcd[np.logical_or(np.logical_or(pcd[:, 0] < aabb.min_x, pcd[:, 0] > aabb.max_x),
                                    np.logical_or(pcd[:, 1] < aabb.min_y, pcd[:, 1] > aabb.max_y))]

        train_pcd = torch.from_numpy(pcd).cuda()
        model_input = self.model.backbone.prepare_input(train_pcd)

        model_input = KittiDataset.collate_batch([model_input])
        for key, val in model_input.items():
            if not isinstance(val, np.ndarray):
                continue
            model_input[key] = torch.from_numpy(val).float().cuda()
        batch_dict = self.model(model_input, metric_head=False, compute_rotation=False, compute_transl=False)
        embedding = batch_dict['out_embedding'][0].cpu().numpy()

        query_id, query_distance = self.index.knn_query(embedding, k=1)
        query_id, query_distance = query_id[0][0], query_distance[0][0]

        coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
        point_features = batch_dict['point_features_NV'].squeeze(-1)
        coords = coords[0]
        point_features = point_features[0]

        train_point_cloud = o3d.geometry.PointCloud()
        train_point_cloud.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy())
        train_pcd_feature = reg_module.Feature()
        train_pcd_feature.data = point_features.permute(0, 1).cpu().numpy()

        query_point_cloud, query_point_features_data = self._get_data_for_frame(query_id)
        query_pcd_feature = reg_module.Feature()
        query_pcd_feature.data = query_point_features_data

        # np.save('pcd_samples/train_pcd.npy', np.asarray(train_point_cloud.points))
        # np.save('pcd_samples/train_features.npy', point_features.permute(0, 1).cpu().numpy())
        # np.save('pcd_samples/query_pcd.npy', np.asarray(query_point_cloud.points))
        # np.save('pcd_samples/query_features.npy', query_point_features_data)
        # sys.exit(1)

        result: o3d.pipelines.registration.RegistrationResult = \
            reg_module.registration_ransac_based_on_feature_matching(
                query_point_cloud, train_point_cloud, query_pcd_feature, train_pcd_feature, True,
                10.0,
                reg_module.TransformationEstimationPointToPoint(False),
                3, [],
                reg_module.RANSACConvergenceCriteria(max_iteration=1000, confidence=1.0))

        if result.fitness < 0.8 or result.inlier_rmse > 1.1:
            return None, None
        query_point_cloud.transform(result.transformation)
        # self.visualizer.add_geometry(train_point_cloud, 'train')
        # self.visualizer.add_geometry(query_point_cloud, 'query')
        # print(f'Fitness: {result.fitness}, rmse: {result.inlier_rmse}, d: {query_distance}',
        #       f'{distance_to_second_query}')
        return self.geo_poses[query_id], result.transformation





