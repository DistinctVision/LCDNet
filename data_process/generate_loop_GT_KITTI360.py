import argparse
import enum
import torch
from torch.utils.data import Dataset
import os
from sklearn.neighbors import KDTree
import pickle
import numpy as np
from tqdm import tqdm


class KITTI360(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, positive_range=5., negative_range=25., hard_range=None):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        self.positive_range = positive_range
        self.negative_range = negative_range
        self.hard_range = hard_range
        self.dir = dir
        self.sequence = sequence
        calib_file = os.path.join(dir, 'calibration', 'calib_cam_to_velo.txt')
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                data = np.array([float(x) for x in line.split()])

        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)

        self.frames_with_gt = []
        poses2 = []
        poses = os.path.join(dir, 'data_poses', sequence, 'cam0_to_world.txt')
        scan_dir = os.path.join(dir, 'data_3d_raw', sequence)
        without_ground = True
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                idx = int(x[0])
                # if without_ground:
                #     velo_path = os.path.join(scan_dir, 'sequences', f'{sequence}', 'velodyne_no_ground', f'{idx:06d}.h5')
                # else:
                #     velo_path = os.path.join(scan_dir, 'sequences', f'{sequence}', 'velodyne', f'{idx:06d}.bin')
                # if not os.path.exists(velo_path):
                #     continue

                self.frames_with_gt.append(idx)
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[1:5])
                pose[1, 0:4] = torch.tensor(x[5:9])
                pose[2, 0:4] = torch.tensor(x[9:13])
                pose[3, 3] = 1.0
                pose = pose @ cam0_to_velo.inverse()
                poses2.append(pose.float().numpy())
        self.frames_with_gt = np.array(self.frames_with_gt, dtype=int)
        poses2 = np.stack(poses2)
        self.poses = poses2
        self.kdtree = KDTree(self.poses[:, :3, 3])

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):

        x = self.poses[idx, 0, 3]
        y = self.poses[idx, 1, 3]
        z = self.poses[idx, 2, 3]

        anchor_pose = torch.tensor([x, y, z])

        indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.positive_range)
        indices = [int(indx) for indx in indices[0]]
        min_range = max(0, idx-50)
        max_range = min(idx+50, self.poses.shape[0])
        positive_idxs = list(set(indices) - set(range(min_range, max_range)))
        positive_idxs.sort()
        num_loop = len(positive_idxs)
        if num_loop > 0:
            positive_idxs = list(self.frames_with_gt[np.array(positive_idxs)])

        indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.negative_range)
        indices = [int(indx) for indx in indices[0]]
        indices = set(indices)
        negative_idxs = set(range(self.poses.shape[0])) - indices
        negative_idxs = list(negative_idxs)
        negative_idxs.sort()

        hard_idxs = None
        if self.hard_range is not None:
            inner_indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.hard_range[0])
            outer_indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.hard_range[1])
            hard_idxs = set(outer_indices[0]) - set(inner_indices[0])
            hard_idxs = list(self.frames_with_gt[np.array(list(hard_idxs))])
            pass

        return num_loop, positive_idxs,\
               list(self.frames_with_gt[np.array(negative_idxs)]),\
               hard_idxs


if __name__ == '__main__':
    from tqdm import tqdm
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', default='./KITTI-360', help='dataset directory')
    args = parser.parse_args()

    sequences = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync",
                 "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0006_sync",
                 "2013_05_28_drive_0007_sync", "2013_05_28_drive_0009_sync", "2013_05_28_drive_0010_sync"]

    base_dir = args.root_folder
    for seq_index, sequence in enumerate(sequences):
        dataset = KITTI360(base_dir, sequence, 4, 10, [6, 10])
        lc_gt = []
        lc_gt_file = os.path.join(base_dir, 'data_poses', sequence, 'loop_GT_4m.pickle')

        progress_bar = tqdm(range(len(dataset)), desc=f'{seq_index+1}/{len(sequences)}')
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
        print(f'Sequence {sequence} done')
