from typing import List, Dict
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

import plotly.graph_objects as go

from localization.global_localization import GlobalLocalizer


def main(dataset_reader: AstralDatasetReader, db_dataset_id: str):

    localizer = GlobalLocalizer(db_dataset_id, location=dataset_reader.location)

    number_of_frames = min(3000, len(dataset_reader))

    output: Dict[str, List[float]] = {
        'delta': [],
        'rotation_error': [],
        'position_error': [],
        'success_rate': []
    }
    with torch.no_grad():
        for i in tqdm(range(3, number_of_frames, 10), 'train'):
            data = dataset_reader[i]

            frame_transform = dataset_reader.transform_manager.get_transform('ld_cc', 'map')
            frame_position = v3d.create(*frame_transform[:3, 3])

            query_transform = localizer.localize(data['ld_cc'], frame_transform)
            output['rotation_error'].append(localizer.last_rotation_error)
            output['position_error'].append(localizer.last_position_error)
            output['success_rate'].append(localizer.last_success_rate)
            if query_transform is None:
                output['delta'].append(-1.0)
                continue
            query_position = v3d.create(*query_transform[:3, 3])
            delta = v3d.length(v3d.sub(frame_position, query_position))
            output['delta'].append(delta)
            print(delta)

    fig = go.Figure()
    size = len(output['delta'])
    for out_name, out_list in output.items():
        fig.add_trace(go.Scatter(x=np.arange(size), y=out_list, mode='lines', name=out_name))
    fig.show()

    localizer.stop()


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
