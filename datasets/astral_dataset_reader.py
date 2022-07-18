from typing import Union, TypeVar, List, Dict
from pathlib import Path
import argparse

from copy import deepcopy

import cv2
import numpy as np
from mldatatools.dataset import Dataset
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
import mldatatools.utils.map as map_tool
from mldatatools.dataset import Frame, Message, Sensor, PointCloudMsg, ImageMsg, \
    GlobalPosition, GNSSMsg, ImuMsg, GNSSMsgData, ImuMsgData, GlobalPoseMsg, GlobalPoseMsgData, Annotation
from torch.utils.data import Dataset as TorchDataset


MessageT = TypeVar('MessageT', bound=Message)
SensorT = TypeVar('SensorT', bound=Sensor)


class AstralDatasetReader(TorchDataset):
    def __init__(self,
                 dataset_dir: Union[Path, str],
                 location: str,
                 sensor_names: List[str]):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.location = location
        self.astral_dataset = Dataset.load(self.dataset_dir, check=False)

        self.sensors = {sensor_name: self._get_sensor_by_name(sensor_name) for sensor_name in sensor_names}
        self.gnss = self._get_sensor_by_name('nmea')
        self.imu = self._get_sensor_by_name('imu')

        self._transform_manager = self.astral_dataset.annotation.tf.manager

    @property
    def annotation(self) -> Annotation:
        return self.astral_dataset.annotation

    @property
    def transform_manager(self) -> TransformManager:
        return self._transform_manager

    def update_transform(self, frame_index: int):
        frame = self.astral_dataset.get_frame(frame_index)
        self._update_transform_for_frame(frame)

    def _update_transform_for_frame(self, frame: Frame) -> GlobalPoseMsgData:
        geo_pose_msg: GlobalPoseMsg = self._get_msg_by_sensor(frame, self.gnss).value
        geo_pose = geo_pose_msg.data
        position = map_tool.gps_to_local((geo_pose.position.latitude,
                                          geo_pose.position.longitude,
                                          geo_pose.position.altitude,), self.location)
        rotation = (geo_pose.orientation.w, geo_pose.orientation.x, geo_pose.orientation.y, geo_pose.orientation.z)
        t = pt.transform_from_pq((*position, *rotation,))
        self._transform_manager.add_transform('gps', 'map', pt.transform_from_pq((*position, *rotation,)),)
        return geo_pose

    def _get_sensor_by_name(self, sensor_name: str) -> SensorT:
        return self.astral_dataset.annotation.sensors[
            [x.name for x in self.astral_dataset.annotation.sensors].index(sensor_name)]

    def _get_msg_by_sensor(self, frame: Frame, sensor: Sensor) -> MessageT:
        return frame.msg_ids[[msg.value.sensor_id for msg in frame.msg_ids].index(sensor.id)]

    def _convert_known_msg(self, msg: Message):
        if isinstance(msg, ImageMsg):
            return cv2.imread(str(self.dataset_dir / msg.data.image_path))
        if isinstance(msg, PointCloudMsg):
            point_cloud = msg.data.load(self.dataset_dir, remove_nan_points=True)
            pcd = np.asarray(point_cloud.points, dtype=float)
            intensities = np.asarray(point_cloud.colors, dtype=float)[:, 0:1]
            pcd = np.append(pcd, intensities, axis=1)
            return pcd
        if isinstance(msg, GNSSMsg):
            return msg.data
        if isinstance(msg, GlobalPoseMsg):
            return msg.data
        if isinstance(msg, ImuMsg):
            return msg.data
        return msg

    def __len__(self):
        return len(self.astral_dataset.frames)

    def __getitem__(self, item_index: int) -> dict:
        frame = self.astral_dataset.get_frame(item_index)
        geo_pose = self._update_transform_for_frame(frame)

        out = {'geo_pose': geo_pose}
        for sensor_name, sensor in self.sensors.items():
            msg = self._get_msg_by_sensor(frame, sensor)
            out[sensor_name] = self._convert_known_msg(msg.value)
        return out


