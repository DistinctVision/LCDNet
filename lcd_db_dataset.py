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

        embeddings = [self._get_embedding_from_frame(frame) for frame in self.astral_dataset.iter_frames()]
        self.geo_poses = [self._get_msg_by_sensor(frame, self.gnss) for frame in self.astral_dataset.iter_frames()]

        indices = [i for i in range(len(embeddings))]

        self.index = hnswlib.Index(space='l2', dim=256)
        self.index.init_index(len(embeddings))
        self.index.add_items(embeddings, indices)

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

    def _get_embedding_from_frame(self, frame: Frame) -> Optional[NumpyAnnotation]:
        for pkg in frame.packages:
            if pkg.class_name == 'embedding':
                return pkg.data
        return None

    def _get_sensor_by_name(self, sensor_name: str) -> SensorT:
        return self.astral_dataset.annotation.sensors[
            [x.name for x in self.astral_dataset.annotation.sensors].index(sensor_name)]

    def _get_msg_by_sensor(self, frame: Frame, sensor: Sensor) -> MessageT:
        return frame.msg_ids[[msg.value.sensor_id for msg in frame.msg_ids].index(sensor.id)]

    def _get_data_for_frame(self, frame_index: int) -> Tuple[o3d.geometry.PointCloud, reg_module.Feature]:
        frame = self.astral_dataset.get_frame(frame_index)
        point_cloud = self._get_msg_by_sensor(frame, self.lidar).data.load(self.astral_dataset.root_dir,
                                                                           remove_nan_points=True)
        point_features: reg_module.Feature = None
        for pkg in frame.packages:
            if pkg.class_name == 'pcd_features':
                point_features = pkg.data.load(self.astral_dataset.root_dir)
        return point_cloud, point_features

    def __getitem__(self, pcd: np.ndarray) -> Tuple[Optional[GlobalPoseMsgData], Optional[np.ndarray]]:
        query_pcd = torch.from_numpy(pcd).cuda()
        model_input = self.model.backbone.prepare_input(query_pcd)

        model_input = KittiDataset.collate_batch([model_input])
        for key, val in model_input.items():
            if not isinstance(val, np.ndarray):
                continue
            model_input[key] = torch.from_numpy(val).float().cuda()
        batch_dict = self.model(model_input, metric_head=False, compute_rotation=False, compute_transl=False)
        embedding = batch_dict['out_embedding'][0].cpu().numpy()

        train_id = self.index.knn_query(embedding)

        coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
        point_features = batch_dict['point_features_NV'].squeeze(-1)
        coords = coords[0]
        point_features = point_features[0]

        query_pcd = o3d.geometry.PointCloud()
        query_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy())
        query_pcd_feature = reg_module.Feature()
        query_pcd_feature.data = point_features.permute(0, 1).cpu().numpy()

        train_point_cloud, train_point_features = self._get_data_for_frame(train_id)

        result: o3d.pipelines.registration.RegistrationResult = \
            reg_module.registration_ransac_based_on_feature_matching(
                query_pcd, train_point_cloud, query_pcd_feature, train_point_features, True,
                0.6,
                reg_module.TransformationEstimationPointToPoint(False),
                3, [],
                reg_module.RANSACConvergenceCriteria(5000))

        if result.fitness < 0.8 or result.inlier_rmse > 0.2:
            return None, None

        return self.geo_poses[train_id], result.transformation





