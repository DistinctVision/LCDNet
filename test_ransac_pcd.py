import math
import sys
from typing import List, Tuple

import random
import time

from tqdm import tqdm
from multiprocessing import Pool, set_start_method
from copy import deepcopy

import os
from sklearn.neighbors import KDTree
from pathlib import Path
import numpy as np
from pytransform3d import transformations as pt

import plotly.graph_objects as go

import open3d as o3d

reg_module = o3d.pipelines.registration

from mldatatools.utils.visualizer3d import Visualizer3d


def arun(points_a: np.ndarray, points_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 3D registration using Arun's method: B = RA + t
    """
    n = points_a.shape[0]
    assert points_b.shape[0] == n

    # calculate centroids
    a_centroid = (np.sum(points_a[:, :3], axis=0)) * 1/n
    b_centroid = (np.sum(points_b[:, :3], axis=0)) * 1/n

    # calculate the vectors from centroids
    a_prime = points_a - a_centroid
    b_prime = points_b - b_centroid

    # rotation estimation
    h = np.zeros([3, 3])
    for i in range(n):
        ai = a_prime[i, :3].T
        bi = b_prime[i, :3].T
        h = h + np.outer(ai, bi)
    u, s, v_transpose = np.linalg.svd(h)
    v = np.transpose(v_transpose)
    u_transpose = np.transpose(u)
    R = v @ np.diag([1, 1, np.linalg.det(v) * np.linalg.det(u_transpose)]) @ u_transpose

    # translation estimation
    b_ = b_centroid.T
    a_ = R @ a_centroid.T
    t = b_centroid.T - R @ a_centroid.T

    return R, t


def generate_random_transform(max_angle: float, max_parallax: float) -> np.ndarray:
    r, t = generate_random_rotation(max_angle), generate_random_translation(max_parallax)
    return to_transform(r, t)


def to_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.identity(4, float)
    m[:3, :3] = r
    m[:3, 3] = t.T
    return m


def generate_random_translation(max_parallax: float) -> np.ndarray:
    return np.random.uniform(-max_parallax, max_parallax, 3)


def generate_random_rotation(max_angle: float) -> np.ndarray:
    rpy = np.random.uniform(-max_angle, max_angle, 3)

    R1 = np.array([[1.0,  0.0,  0.0],
                   [0.0,  np.cos(rpy[0]), -np.sin(rpy[0])],
                   [0.0,  np.sin(rpy[0]),  np.cos(rpy[0])]])

    R2 = np.array([[np.cos(rpy[1]),  0.0,  np.sin(rpy[1])],
                   [0.0,  1.0,  0.0],
                   [-np.sin(rpy[1]),  0.0,  np.cos(rpy[1])]])

    R3 = np.array([[np.cos(rpy[2]), -np.sin(rpy[2]),  0.0],
                   [np.sin(rpy[2]),  np.cos(rpy[2]),  0.0],
                   [0.0,  0.0,  1.0]])

    return R3.dot(R2.dot(R1))


def _ransac_star_(args) -> dict:
    random.seed(args[0] + args[1])
    np.random.seed(args[0] + args[1])
    return _ransac_(*args[2:])


def _ransac_(query_points: np.ndarray,
             train_points: np.ndarray,
             query_kdtree,
             search_radius: float,
             pairs: List[Tuple[int, int]],
             number_of_iterations: int) -> dict:
    #query_point_cloud = o3d.geometry.PointCloud()
    #query_point_cloud.points = o3d.utility.Vector3dVector(query_points[:, :3])
    #train_point_cloud = o3d.geometry.PointCloud()
    #train_point_cloud.points = o3d.utility.Vector3dVector(train_points[:, :3])
    train_point = np.hstack((
        train_points,
        np.ones((train_points.shape[0], 1))
    ))

    np.random.shuffle(pairs)
    best_inliers = None
    best_count = -1
    best_transform = np.identity(4, float)
    for _ in range(number_of_iterations):
        for i in range(3):
            r_index = random.randint(0, len(pairs) - 1)
            if r_index != i:
                pairs[i], pairs[r_index] = pairs[r_index], pairs[i]
        # pairs3 = pairs[np.random.choice(len(pairs), 3, replace=False)]
        points_a = query_points[pairs[:3, 0]]
        points_b = train_points[pairs[:3, 1]]
        transform = to_transform(*arun(points_a, points_b))
        transform = np.linalg.inv(transform)
        new_train_point = (transform @ train_point.T)[:3].T
        count = query_kdtree.query_radius(new_train_point, search_radius, count_only=True).sum()
        if count > best_count:
            #best_inliers = inliers
            best_count = count
            best_transform = transform
    return {
        # 'inliers': best_inliers,
        'count': best_count,
        'transform': best_transform
    }


class Ransac:
    def __init__(self,
                 query_points: np.ndarray, query_features: np.ndarray,
                 train_points: np.ndarray, train_features: np.ndarray):
        assert query_points.shape[0] == query_features.shape[0], 'Mismatch number of query points and features'
        assert train_points.shape[0] == train_features.shape[0], 'Mismatch number of train points and features'

        self.query_points = query_points
        self.query_kdtree = KDTree(query_points, leaf_size=100)
        self.query_features = query_features
        self.train_points = train_points
        self.train_features = train_features

        self.number_of_iterations = 1500
        self.search_radius = 0.5
        self.cosine_similarity_threshold = 0.85

        self.pairs = self._make_pairs(10.0)

    def _make_pairs(self, max_pair_distance: float) -> List[Tuple[int, int]]:
        max_pair_distance_sq = max_pair_distance * max_pair_distance
        pairs = []
        query_feature_lengths = np.linalg.norm(self.query_features, axis=1)
        train_feature_lengths = np.linalg.norm(self.train_features, axis=1)
        for query_index, (query_point, query_feature, query_length) in enumerate(zip(self.query_points,
                                                                                     self.query_features,
                                                                                     query_feature_lengths)):
            # delta = np.subtract(self.train_points, query_point)
            # sq_distances = (delta * delta).sum(axis=1)
            # near_idxs = np.where(sq_distances < max_pair_distance_sq)
            # if near_idxs[0].shape[0] < 1:
            #     continue
            cosine_similarity = np.divide(np.dot(self.train_features, query_feature),
                                          np.sqrt(query_length * train_feature_lengths))
            best_match = np.argmax(cosine_similarity)
            max_cosine_similarity = cosine_similarity[best_match]
            # best_match = near_idxs[0][best_match]
            if max_cosine_similarity > self.cosine_similarity_threshold:
                pairs.append((query_index, best_match, max_cosine_similarity))
        for train_index, (train_point, train_feature, train_length) in enumerate(zip(self.train_points,
                                                                                     self.train_features,
                                                                                     train_feature_lengths)):
            # delta = np.subtract(self.query_points, train_point)
            # sq_distances = (delta * delta).sum(axis=1)
            # near_idxs = np.where(sq_distances < max_pair_distance_sq)
            # if near_idxs[0].shape[0] < 1:
            #     continue
            cosine_similarity = np.divide(np.dot(self.query_features, train_feature),
                                          np.sqrt(train_length * query_feature_lengths))
            best_match = np.argmax(cosine_similarity, axis=0)
            max_cosine_similarity = cosine_similarity[best_match]
            # best_match = near_idxs[0][best_match]
            if max_cosine_similarity > self.cosine_similarity_threshold:
                pairs.append((best_match, train_index, max_cosine_similarity))

        points_a = self.query_points[:, :3].take([q_idx for q_idx, t_idx, sv in pairs], axis=0)
        points_b = self.train_points[:, :3].take([t_idx for q_idx, t_idx, sv in pairs], axis=0)
        delta = np.subtract(points_b, points_a)
        sq_distances = (delta * delta).sum(axis=1)

        pairs = np.asarray(pairs)[sq_distances < (max_pair_distance ** 2)]
        # pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        # pairs = pairs[:len(pairs)//2]

        return pairs[:, :2].astype(int)

    def run(self, number_of_workers: int = 1):
        if number_of_workers > 1:
            common_random_seed = int(time.time_ns() * 1e6) % 100000
            part_number_of_iterations = int(math.ceil(self.number_of_iterations / number_of_workers))
            with Pool(number_of_workers) as pool:
                parts = pool.map(_ransac_star_,
                                 ((common_random_seed, worker_index,
                                   self.query_points, self.train_points, self.query_kdtree,
                                   self.search_radius, self.pairs, part_number_of_iterations)
                                  for worker_index in range(number_of_workers)), chunksize=1)
            best_part = None
            for part in parts:
                if best_part is None:
                    best_part = part
                if part['count'] > best_part['count']:
                    best_part = part
        else:
            best_part = _ransac_(self.query_points, self.train_points, self.query_kdtree, self.search_radius, self.pairs,
                                 self.number_of_iterations)

        pcd_size = min(self.query_points.shape[0], self.train_points.shape[0])
        # print(f'best_count: {best_count}/{pcd_size}, best_transform: {best_transform}')
        return best_part['transform'], best_part['count'] / pcd_size


def test_arun(query_pcd: np.ndarray, train_pcd: np.ndarray):
    np.random.shuffle(query_pcd)
    a = pt.vectors_to_points(query_pcd[:, :3])
    r, t = generate_random_rotation(0.1), generate_random_translation(0.3)
    transform = to_transform(r, t)
    b = np.dot(transform, a.T).T
    r1, t1 = arun(a[:, :3], b[:, :3])
    transform1 = to_transform(r1, t1)
    sys.exit(1)


def test():
    #set_start_method('forkserver')

    query_pcd = np.load(str(Path('pcd_samples') / 'query_pcd.npy'))
    query_feature = np.load(str(Path('pcd_samples') / 'query_features.npy')).T
    train_pcd = np.load(str(Path('pcd_samples') / 'train_pcd.npy'))
    train_feature = np.load(str(Path('pcd_samples') / 'train_features.npy')).T

    gt_transform = np.asarray([[9.99995730e-01,  2.84442393e-03, - 6.70004790e-04, - 3.15555224e+00],
                               [-2.84688719e-03,  9.99989085e-01, - 3.70467126e-03, - 2.44902985e-01],
                               [6.59459821e-04, 3.70656286e-03, 9.99992913e-01, - 9.31968256e-02],
                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    gt_position = - np.dot(gt_transform[:3, :3].T, gt_transform[:3, 3])

    output = {'success': [], 'coeff': [], 'rotation_error': [], 'position_error': []}
    avg_time = 0
    valid_counter = 0
    progress_bar = tqdm(range(100), 'Test custom ransac')
    for _ in progress_bar:
        t0 = time.time()
        ransac = Ransac(query_pcd[:, :3], query_feature, train_pcd[:, :3], train_feature)
        r_transform, coeff = ransac.run(12)
        avg_time += time.time() - t0
        r_position = - np.dot(r_transform[:3, :3].T, r_transform[:3, 3])
        rotation_error = np.max(np.abs(gt_transform[:3, :3] - r_transform[:3, :3]))
        position_error = np.max(np.abs(gt_position - r_position))
        output['coeff'].append(coeff)
        output['rotation_error'].append(rotation_error)
        output['position_error'].append(position_error)
        if rotation_error < 0.015 and position_error < 0.7:
            valid_counter += 1
            progress_bar.set_description(f'Test custom ransac: {valid_counter}')
            output['success'].append(1)
        else:
            output['success'].append(0)
    avg_time /= 100
    print(f'Ransac: {valid_counter} / 100, average time: {avg_time}')

    fig = go.Figure()
    size = len(output['coeff'])
    for out_name, out_list in output.items():
        fig.add_trace(go.Scatter(x=np.arange(size), y=out_list, mode='lines', name=out_name))
    fig.show()

    avg_time = 0
    valid_counter = 0
    progress_bar = tqdm(range(100), 'Test Open3D ransac')
    for _ in progress_bar:
        t0 = time.time()

        query_point_cloud = o3d.geometry.PointCloud()
        query_point_cloud.points = o3d.utility.Vector3dVector(query_pcd[:, :3])
        train_point_cloud = o3d.geometry.PointCloud()
        train_point_cloud.points = o3d.utility.Vector3dVector(train_pcd[:, :3])
        train_point_cloud.colors = o3d.utility.Vector3dVector([(1, 0, 0) for _ in range(train_pcd.shape[0])])

        query_pcd_feature = reg_module.Feature()
        query_pcd_feature.data = query_feature.T
        train_pcd_feature = reg_module.Feature()
        train_pcd_feature.data = train_feature.T

        result: o3d.pipelines.registration.RegistrationResult = \
            reg_module.registration_ransac_based_on_feature_matching(
                query_point_cloud, train_point_cloud, query_pcd_feature, train_pcd_feature, True,
                10.0,
                reg_module.TransformationEstimationPointToPoint(False),
                3, [],
                reg_module.RANSACConvergenceCriteria(max_iteration=1000, confidence=1.0))
        avg_time += time.time() - t0
        r_transform = np.linalg.inv(result.transformation)
        r_position = - np.dot(r_transform[:3, :3].T, r_transform[:3, 3])
        rotation_error = np.max(np.abs(gt_transform[:3, :3] - r_transform[:3, :3]))
        position_error = np.max(np.abs(gt_position - r_position))
        if rotation_error < 0.015 and position_error < 0.7:
            valid_counter += 1
            progress_bar.set_description(f'Test Open3D ransac: {valid_counter}')
    avg_time /= 100
    print(f'Ransac: {valid_counter} / 100, average time: {avg_time}')

    train_point_cloud.transform(gt_transform)

    visualizer = Visualizer3d()
    visualizer.start()

    visualizer.add_geometry(query_point_cloud, 'query')
    visualizer.add_geometry(train_point_cloud, 'train')


if __name__ == '__main__':
    test()
