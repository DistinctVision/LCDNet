from mldatatools.utils.visualizer3d import Visualizer3d

from pathlib import Path
import numpy as np

import open3d as o3d
reg_module = o3d.pipelines.registration


def test():
    query_pcd = np.load(str(Path('pcd_samples') / 'query_pcd.npy'))
    query_feature = np.load(str(Path('pcd_samples') / 'query_features.npy'))
    train_pcd = np.load(str(Path('pcd_samples') / 'train_pcd.npy'))
    train_feature = np.load(str(Path('pcd_samples') / 'train_features.npy'))

    query_point_cloud = o3d.geometry.PointCloud()
    query_point_cloud.points = o3d.utility.Vector3dVector(query_pcd[:, :3])
    train_point_cloud = o3d.geometry.PointCloud()
    train_point_cloud.points = o3d.utility.Vector3dVector(train_pcd[:, :3])
    train_point_cloud.colors = o3d.utility.Vector3dVector([(1, 0, 0) for _ in range(train_pcd.shape[0])])

    query_pcd_feature = reg_module.Feature()
    query_pcd_feature.data = query_feature.transpose((0, 1,))
    train_pcd_feature = reg_module.Feature()
    train_pcd_feature.data = train_feature.transpose((0, 1,))

    result: o3d.pipelines.registration.RegistrationResult = \
        reg_module.registration_ransac_based_on_feature_matching(
            query_point_cloud, train_point_cloud, query_pcd_feature, train_pcd_feature, True,
            10.0,
            reg_module.TransformationEstimationPointToPoint(False),
            3, [],
            reg_module.RANSACConvergenceCriteria(10000))

    query_point_cloud.transform(result.transformation)
    print(result)

    visualizer = Visualizer3d()
    visualizer.start()

    visualizer.add_geometry(train_point_cloud, 'train')
    visualizer.add_geometry(query_point_cloud, 'query')


if __name__ == '__main__':
    test()
