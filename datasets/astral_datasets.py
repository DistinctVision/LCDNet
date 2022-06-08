from typing import Union, TypeVar
from pathlib import Path
import argparse

import numpy as np
import torch
import random
from mldatatools.dataset import Dataset
from pytransform3d import transformations as pt
import mldatatools.utils.map as map_tool
from mldatatools.dataset import Frame, Message, Sensor, PointCloudMsg
from torch.utils.data import Dataset as TorchDataset
import pickle


MessageT = TypeVar('MessageT', bound=Message)


class AstralPoses(TorchDataset):
    def __init__(self,
                 dataset_dir: Union[Path, str],
                 sensor_name: str,
                 location: str,
                 train: bool = True,
                 loop_file: str = 'loop_GT',
                 jitter: bool = False,
                 remove_random_angle: float = -1):
        super(AstralPoses, self).__init__()

        self.train = train
        self.dataset_dir = Path(dataset_dir)
        self.jitter = jitter
        self.remove_random_angle = remove_random_angle

        self.sensor_name = sensor_name
        self.astral_dataset = Dataset.load(dataset_dir)

        self.sensor = self._get_sensor_by_name(sensor_name)
        gnss = self._get_sensor_by_name('gnss')
        imu = self._get_sensor_by_name('imu')

        # Prepare positions
        positions = (self._get_msg_by_sensor(frame, gnss).data.position
                     for frame in self.astral_dataset.iter_frames(follow_id=True))
        positions = ((position.latitude, position.longitude, position.altitude,) for position in positions)
        positions = [map_tool.gps_to_local(position, location=location) for position in positions]

        # Prepare rotations
        rotations = (self._get_msg_by_sensor(frame, imu).data.orientation
                     for frame in self.astral_dataset.iter_frames(follow_id=True))
        rotations = [(rotation.orientation.w, rotation.orientation.x, rotation.orientation.y, rotation.orientation.z,)
                     for rotation in rotations]

        tm = self.astral_dataset.annotation.tf.manager
        poses = []

        for frame_idx in range(len(self.astral_dataset.frames)):
            position = positions[frame_idx]
            rotation = rotations[frame_idx]
            tm.add_transform('gps', 'map', pt.transform_from_pq((*position, *rotation,)),)
            sensor_transform = tm.get_transform(self.sensor_name, 'map')
            poses.append(sensor_transform)
        self.poses = poses

        gt_file = self.dataset_dir / f'{loop_file}.pickle'
        self.loop_gt = []
        with open(gt_file, 'rb') as f:
            temp = pickle.load(f)
            for elem in temp:
                temp_dict = {'idx': elem['idx'], 'positive_idxs': elem['positive_idxs']}
                self.loop_gt.append(temp_dict)
            del temp
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

    def _get_sensor_by_name(self, sensor_name: str) -> Sensor:
        return self.astral_dataset.annotation.sensors[
            [x.name for x in self.astral_dataset.annotation.sensors].index(sensor_name)]

    def _get_msg_by_sensor(self, frame: Frame, sensor: Sensor) -> MessageT:
        return frame.msg_ids[[msg.sensor_id for msg in frame.msg_ids].index(sensor.id)]

    def _load_pcd(self, frame_idx: int, sensor: Sensor) -> np.ndarray:
        frame = self.astral_dataset.get_frame(frame_idx, follow_id=True)
        point_cloud_msg: PointCloudMsg = self._get_msg_by_sensor(frame, sensor)
        point_cloud = point_cloud_msg.data.load(self.dataset_dir, remove_nan_points=True)
        pcd = np.asarray(point_cloud.points, dtype=float)
        intensities = np.asarray(point_cloud.colors, dtype=float)[:, 0:1]
        pcd = np.append(pcd, intensities, axis=1)
        return pcd

    def _do_pcd_augmentation(self, pcd: np.ndarray) -> np.ndarray:
        if self.jitter:
            noise = 0.01 * np.random.randn(pcd.shape[0], pcd.shape[1]).astype(np.float32)
            noise = np.clip(noise, -0.05, 0.05)
            pcd = pcd + noise

        if self.remove_random_angle > 0:
            azi = np.arctan2(pcd[..., 1], pcd[..., 0])
            cols = 2084 * (np.pi - azi) / (2 * np.pi)
            cols = np.minimum(cols, 2084 - 1)
            cols = np.int32(cols)
            start_idx = np.random.randint(0, 2084)
            end_idx = start_idx + (self.remove_random_angle / (360.0/2084))
            end_idx = int(end_idx % 2084)
            remove_idxs = cols > start_idx
            remove_idxs = remove_idxs & (cols < end_idx)
            pcd = pcd[np.logical_not(remove_idxs)]
        return pcd

    def __len__(self):
        return len(self.astral_dataset.frames)

    def __getitem__(self, frame_idx: int):
        anchor_pcd = self._load_pcd(frame_idx, self.sensor)
        anchor_pcd = self._do_pcd_augmentation(anchor_pcd)

        if self.train:
            x = self.poses[frame_idx][0, 3]
            y = self.poses[frame_idx][1, 3]
            z = self.poses[frame_idx][2, 3]

            anchor_pose = torch.tensor([x, y, z])
            possible_match_pose = torch.tensor([0., 0., 0.])
            negative_pose = torch.tensor([0., 0., 0.])

            indices = list(range(len(self.poses)))
            cont = 0
            positive_idx = frame_idx
            negative_idx = frame_idx
            while cont < 2:
                i = random.choice(indices)
                possible_match_pose[0] = self.poses[frame_idx][0, 3]
                possible_match_pose[1] = self.poses[frame_idx][1, 3]
                possible_match_pose[2] = self.poses[frame_idx][2, 3]
                distance = torch.norm(anchor_pose - possible_match_pose)
                if distance <= 4 and frame_idx == positive_idx:
                    positive_idx = i
                    cont += 1
                elif distance > 10 and frame_idx == negative_idx:
                    negative_idx = i
                    cont += 1

            positive_pcd = self._load_pcd(positive_idx, self.sensor)
            positive_pcd = self._do_pcd_augmentation(positive_pcd)
            negative_pcd = self._load_pcd(negative_idx, self.sensor)
            negative_pcd = self._do_pcd_augmentation(negative_pcd)

            sample = {'anchor': torch.from_numpy(anchor_pcd),
                      'positive': torch.from_numpy(positive_pcd),
                      'negative': torch.from_numpy(negative_pcd)}
        else:
            sample = {'anchor': torch.from_numpy(anchor_pcd)}

        return sample
