from typing import Union, Optional, List, Tuple, Dict
from pathlib import Path

import open3d as o3d
reg_module = o3d.pipelines.registration

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data

from copy import deepcopy

import clearml

from tqdm import tqdm
from datasets.astral_dataset_reader import AstralDatasetReader
from pytransform3d import transformations as pt

import mldatatools.utils.map as map_tool
from mldatatools.utils import v3d
from mldatatools.utils.visualizer3d import Visualizer3d

import plotly.graph_objects as go

from lcd_db_dataset import LcdDbDataset


def main(dataset_reader: AstralDatasetReader, db_dataset_id: str):

    lcd_db = LcdDbDataset(db_dataset_id, location=dataset_reader.location)

    db_transform_manager = deepcopy(dataset_reader.transform_manager)

    number_of_frames = min(3000, len(dataset_reader))

    output: Dict[str, List[float]] = {
            'distance': []
    }
    with torch.no_grad():
        for i in tqdm(range(number_of_frames), 'train'):
            data = dataset_reader[i]

            frame_transform = dataset_reader.transform_manager.get_transform('ld_cc', 'map')
            position = v3d.create(frame_transform[0, 3],
                                  frame_transform[1, 3],
                                  frame_transform[2, 3])

            (geo_pose, delta_transform) = lcd_db(data['ld_cc'], i)
            if geo_pose is None:
                continue
            geo_position = map_tool.gps_to_local((geo_pose.position.latitude,
                                                  geo_pose.position.longitude,
                                                  geo_pose.position.altitude,), dataset_reader.location)
            geo_rotation = (geo_pose.orientation.w, geo_pose.orientation.x,
                            geo_pose.orientation.y, geo_pose.orientation.z)
            db_transform_manager.add_transform('gps', 'map', pt.transform_from_pq((*geo_position, *geo_rotation,)),)

            query_frame_transform = db_transform_manager.get_transform('ld_cc', 'map')
            query_position = v3d.create(query_frame_transform[0, 3],
                                        query_frame_transform[1, 3],
                                        query_frame_transform[2, 3])
            query_frame_transform = np.dot(query_frame_transform, np.linalg.inv(delta_transform))
            query_position_1 = v3d.create(query_frame_transform[0, 3],
                                          query_frame_transform[1, 3],
                                          query_frame_transform[2, 3])

            distance = v3d.length(v3d.sub(query_position, position))
            distance_1 = v3d.length(v3d.sub(query_position_1, position))

            print(f'Found: {distance} {distance_1}')
            output['distance'].append(distance)

    fig = go.Figure()
    size = len(output['distance'])
    for out_name, out_list in output.items():
        fig.add_trace(go.Scatter(x=np.arange(size), y=out_list, mode='lines', name=out_name))
    fig.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='', help="Path to dataset")
    parser.add_argument('--dataset_id', default='', help='Dataset id')
    parser.add_argument('--db_dataset_id', default='', required=True)
    parser.add_argument('--location', default='m11', help='A location for a GNSS data')
    args = parser.parse_args()

    if args.dataset_id:
        clearml_dataset = clearml.Dataset.get(dataset_id=args.dataset_id)
        dataset_path = Path(clearml_dataset.get_local_copy())
    elif args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        raise 'Dataset is not set - "dataset_path" or "dataset_id" argument is needed'

    dataset = AstralDatasetReader(dataset_path, args.location, ['ld_cc'])

    main(dataset, args.db_dataset_id)
