import sys
from typing import Union, Tuple, Optional
from pathlib import Path

import math

import rosservice
import rosgraph
import socket
import os
import threading

import numpy as np
import open3d as o3d
reg_module = o3d.pipelines.registration

from mldatatools.dataset import GlobalPoseMsgData
from pytransform3d import transformations as pt
import pytransform3d.rotations as rpt
import mldatatools.utils.map as map_tool
from mldatatools.utils import v3d
from localization.lcd_db_dataset import LcdDbDataset


def _ros_main_(data: dict):
    stream = os.popen('roslaunch lidar_apollo_instance_segmentation debug_lidar_apollo_instance_segmentation.launch')
    while (line := stream.readline()) and not data['stop']:
        print(line)
    stream.close()


class GlobalLocalizer:
    def __init__(self,
                 dataset_id: Union[Path, str],
                 location: str,
                 rotation_threshold: float = 5.0,  # in degrees
                 position_threshold: float = 0.5,  # in meters
                 min_number_of_success_nodes: int = 3,
                 success_rate_threshold: float = 0.7):
        self._ros_thread = None
        self._ros_thread_data = {'stop': False }
        if not self._check_apollo_service():
            print('Apollo service is not started. Starting...')
            self._star_apollo_service()
        else:
            print('Apollo service is found')

        self.db = LcdDbDataset(dataset_id, location)

        self.step_distance = 2.0
        self.search_distance = 30.0
        self.rotation_threshold = rotation_threshold
        self.position_threshold = position_threshold
        self.min_number_of_success_nodes = min_number_of_success_nodes
        self.success_rate_threshold = success_rate_threshold
        self._actual_frames = []  # [{'query_pose': np.ndarray, 'odom_pose': np.ndarray, 'timestamp': int }]
        self._last_rotation_error = -1.0
        self._last_position_error = -1.0
        self._last_success_rate = -1.0

    def stop(self):
        if self._ros_thread:
            self._ros_thread_data['stop'] = True
            self._ros_thread.join(timeout=5)

    def _star_apollo_service(self):
        self._ros_thread = threading.Thread(target=_ros_main_, args=(self._ros_thread_data,))
        self._ros_thread.start()

    @staticmethod
    def _check_apollo_service() -> bool:
        try:
            rosgraph.Master('/rostopic').getPid()
        except socket.error:
            return False

        service_list = rosservice.get_service_list()
        return service_list.index('/lidar_apollo_instance_segmentation/dynamic_objects') >= 0

    @staticmethod
    def _get_position_from_tf(transform: np.ndarray) -> v3d.Vector3d:
        return v3d.create(*transform[:3, 3])

    def _filter_far_frames(self, odom_position: v3d.Vector3d) -> float:
        filtered_frames = []
        nearest_distance = sys.float_info.max
        for actual_frame in self._actual_frames:
            distance = v3d.length(v3d.sub(self._get_position_from_tf(actual_frame['odom_pose']), odom_position))
            if distance < self.search_distance:
                filtered_frames.append(actual_frame)
                if distance < nearest_distance:
                    nearest_distance = distance
        self._actual_frames = filtered_frames
        return nearest_distance

    def _compare_frame_pose(self, query_pose: np.ndarray, odom_pose: np.ndarray) -> Tuple[int, int]:
        self._last_rotation_error, self._last_position_error = 0.0, 0.0
        n_success, n_fail = 0, 0
        for actual_frame in self._actual_frames:
            delta_query_pose = np.dot(query_pose, np.linalg.inv(actual_frame['query_pose']))
            delta_odom_pose = np.dot(odom_pose, np.linalg.inv(actual_frame['odom_pose']))
            delta_query_odom_pose = np.dot(delta_query_pose, np.linalg.inv(delta_odom_pose))
            _, _, _, delta_angle = rpt.axis_angle_from_matrix(delta_query_odom_pose[:3, :3])
            delta_angle *= 180.0 / math.pi
            delta_position = v3d.length(self._get_position_from_tf(delta_odom_pose))
            if delta_angle < self.rotation_threshold and delta_position < self.position_threshold:
                self._last_rotation_error += delta_angle
                self._last_position_error += delta_position
                n_success += 1
            else:
                n_fail += 1
        if n_success > 0:
            self._last_rotation_error /= n_success
            self._last_position_error /= n_success
            self._last_success_rate = n_success / (n_success + n_fail)
        else:
            self._last_rotation_error, self._last_position_error = -1.0, -1.0
            self._last_success_rate = -1.0
        return n_success, n_fail

    @property
    def last_rotation_error(self) -> float:
        return self._last_rotation_error

    @property
    def last_position_error(self) -> float:
        return self._last_position_error

    @property
    def last_success_rate(self) -> float:
        return self._last_success_rate

    def localize(self, pcd: np.ndarray, odom_pose: np.ndarray, timestamp: int = -1) -> Optional[np.ndarray]:
        nearest_distance = self._filter_far_frames(self._get_position_from_tf(odom_pose))
        if nearest_distance < self.step_distance:
            return None

        geo_pose, delta_transform = self.db(pcd)
        if geo_pose is None or delta_transform is None:
            return None
        geo_position = map_tool.gps_to_local((geo_pose.position.latitude,
                                              geo_pose.position.longitude,
                                              geo_pose.position.altitude,), self.db.location)
        geo_rotation = (geo_pose.orientation.w, geo_pose.orientation.x,
                        geo_pose.orientation.y, geo_pose.orientation.z)
        query_pose = pt.transform_from_pq((*geo_position, *geo_rotation,))
        query_pose = np.dot(query_pose, delta_transform)

        n_success, n_fail = self._compare_frame_pose(query_pose, odom_pose)
        success_rate = n_success / (n_success + n_fail) if (n_success + n_fail) else -1.0

        self._actual_frames.append({
            'query_pose': query_pose,
            'odom_pose': odom_pose,
            'timestamp': timestamp
        })

        if n_success < self.min_number_of_success_nodes or success_rate < self.success_rate_threshold:
            return None
        return query_pose
