from typing import Optional

import rospy

import numpy as np

import starline_msgs.srv


class NdtRosWrapper:
    def __init__(self):
        self._align_service: Optional[rospy.ServiceProxy] = None
        self._ros_connect()

    def _ros_connect(self):
        rospy.wait_for_service('/ndt/align')
        self._align_service = rospy.ServiceProxy('/ndt/align', starline_msgs.srv.AlignPointClouds)

    def align(self, query_pcd: np.ndarray, train_pcd: np.ndarray, guess_transform: np.ndarray = np.identity(4, float)):
        pass