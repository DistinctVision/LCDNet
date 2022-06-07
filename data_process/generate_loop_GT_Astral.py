from typing import Union, Tuple, Optional
from pathlib import Path
import argparse
import torch
from mldatatools.dataset import Dataset
from pytransform3d import transformations as pt
import mldatatools.utils.map as map_tool
from torch.utils.data import Dataset as TorchDataset
import os
from sklearn.neighbors import KDTree
import pickle
import numpy as np

import clearml
from tqdm import tqdm


class GtIndicesAstralDataset(TorchDataset):
    def __init__(self,
                 dataset_dir: Union[Path, str],
                 sensor_name: str,
                 location: str,
                 positive_range: float = 5.,
                 negative_range: float = 25.,
                 hard_range: Optional[Tuple[float, float]] = None):

        self.positive_range = positive_range
        self.negative_range = negative_range
        self.hard_range = hard_range
        self.sensor_name = sensor_name
        self.dataset = Dataset.load(dataset_dir)

        def get_sensor_with_name(t_sensor_name: str):
            sensor = self.dataset.annotation.sensors[[x.name
                                                      for x in self.dataset.annotation.sensors].index(t_sensor_name)]
            return sensor

        def get_msg_with_sensor(messages_list, sensor):
            msg = messages_list[[msg.sensor_id for msg in messages_list].index(sensor.id)]
            return msg

        self.sensor = get_sensor_with_name(sensor_name)
        gnss = get_sensor_with_name('gnss')
        imu = get_sensor_with_name('imu')

        # Prepare positions
        positions = (get_msg_with_sensor(frame.msg_ids, gnss).data.position
                     for frame in self.dataset.iter_frames(follow_id=True))
        positions = ((position.latitude, position.longitude, position.altitude,) for position in positions)
        positions = [map_tool.gps_to_local(position, location=location) for position in positions]

        # Prepare rotations
        rotations = (get_msg_with_sensor(frame.msg_ids, imu).data.orientation
                     for frame in self.dataset.iter_frames(follow_id=True))
        rotations = [(rotation.orientation.w, rotation.orientation.x, rotation.orientation.y, rotation.orientation.z,)
                     for rotation in rotations]

        self.frames_with_gt = []

        tm = self.dataset.annotation.tf.manager
        print(self.dataset.annotation.tf)
        poses = []

        for frame_idx in range(len(self.dataset.frames)):
            frame = self.dataset.get_frame(frame_idx, follow_id=True)
            position = positions[frame_idx]
            rotation = rotations[frame_idx]
            tm.add_transform('gps', 'map', pt.transform_from_pq((*position, *rotation,)),)
            sensor_transform = tm.get_transform(self.sensor_name, 'map')
            poses.append(sensor_transform)
            self.frames_with_gt.append(frame_idx)

        self.frames_with_gt = np.array(self.frames_with_gt, dtype=int)
        poses = np.stack(poses)
        self.poses = poses
        self.kdtree = KDTree(self.poses[:, :3, 3])

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):

        x = self.poses[idx, 0, 3]
        y = self.poses[idx, 1, 3]
        z = self.poses[idx, 2, 3]

        anchor_pose = torch.tensor([x, y, z]).unsqueeze(0).numpy()

        indices = self.kdtree.query_radius(anchor_pose, self.positive_range)
        indices = [int(indx) for indx in indices[0]]
        min_range = max(0, idx - 50)
        max_range = min(idx + 50, self.poses.shape[0])
        positive_idxs = list(set(indices) - set(range(min_range, max_range)))
        positive_idxs.sort()
        num_loop = len(positive_idxs)
        if num_loop > 0:
            positive_idxs = list(self.frames_with_gt[np.array(positive_idxs)])

        indices = self.kdtree.query_radius(anchor_pose, self.negative_range)
        indices = [int(indx) for indx in indices[0]]
        indices = set(indices)
        negative_idxs = set(range(self.poses.shape[0])) - indices
        negative_idxs = list(negative_idxs)
        negative_idxs.sort()

        hard_idxs = None
        if self.hard_range is not None:
            inner_indices = self.kdtree.query_radius(anchor_pose, self.hard_range[0])
            outer_indices = self.kdtree.query_radius(anchor_pose, self.hard_range[1])
            hard_idxs = set(outer_indices[0]) - set(inner_indices[0])
            hard_idxs = list(self.frames_with_gt[np.array(list(hard_idxs))])
            pass

        return num_loop, positive_idxs, \
               list(self.frames_with_gt[np.array(negative_idxs)]), \
               hard_idxs


if __name__ == '__main__':
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', default='', help='Dataset id')
    parser.add_argument('--dataset_path', default='', help='Dataset id')
    args = parser.parse_args()

    if args.dataset_id:
        clearml_dataset = clearml.Dataset.get(dataset_id=args.dataset_id)
        dataset_path = Path(clearml_dataset.get_local_copy())
    elif args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        raise 'Dataset is not set'
    dataset = GtIndicesAstralDataset(dataset_path, 'ld_cc', 'smirnova', 4, 10, (6, 10))

    lc_gt = []
    lc_gt_file = dataset_path / 'loop_GT_4m.pickle'

    progress_bar = tqdm(range(len(dataset)), desc=f'Triples generating')
    for i in progress_bar:
        sample, pos, neg, hard = dataset[i]
        if sample > 0.:
            idx = dataset.frames_with_gt[i]
            sample_dict = {}
            sample_dict['idx'] = int(idx)
            sample_dict['positive_idxs'] = [int(v) for v in pos]
            sample_dict['negative_idxs'] = [int(v) for v in neg]
            sample_dict['hard_idxs'] = [int(v) for v in hard]
            lc_gt.append(sample_dict)
    with open(lc_gt_file, 'wb') as f:
        pickle.dump(lc_gt, f)
    print(f'Output: {dataset_path}')
