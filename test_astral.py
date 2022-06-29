import argparse
import os
from typing import Union, Optional, List
from pathlib import Path
import time
from collections import OrderedDict

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import random

import clearml

from tqdm import tqdm
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from datasets.astral_dataset_reader import AstralDatasetReader

from models.get_models import get_model

import cv2
import v3d

import pandas as pd
import plotly.express as px


EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2 ** 32 - 1)
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(dataset: AstralDatasetReader, weights_path: Union[Path, str]):

    saved_params = torch.load(weights_path, map_location='cpu')

    exp_cfg = saved_params['config']
    exp_cfg['batch_size'] = 6
    exp_cfg['loop_file'] = 'loop_GT_4m'
    exp_cfg['head'] = 'UOTHead'
    exp_cfg['PC_RANGE'] = [-70.4, -70.4, -1, 70.4, 70.4, 3]

    model = get_model(exp_cfg, is_training=False)
    renamed_dict = OrderedDict()
    for key in saved_params['state_dict']:
        if not key.startswith('module'):
            renamed_dict = saved_params['state_dict']
            break
        else:
            renamed_dict[key[7:]] = saved_params['state_dict'][key]

    # Convert shape from old OpenPCDet
    if renamed_dict['backbone.backbone.conv_input.0.weight'].shape != \
            model.state_dict()['backbone.backbone.conv_input.0.weight'].shape:
        for key in renamed_dict:
            if key.startswith('backbone.backbone.conv') and key.endswith('weight'):
                if len(renamed_dict[key].shape) == 5:
                    renamed_dict[key] = renamed_dict[key].permute(-1, 0, 1, 2, 3)

    res = model.load_state_dict(renamed_dict, strict=True)

    if len(res[0]) > 0:
        print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

    model = model.to('cuda:0')
    model.eval()

    train_positions: List[v3d.Vector3d] = []
    train_embeddings: List[np.ndarray] = []

    prev_position: Optional[v3d.Vector3d] = None

    number_of_frames = 1000
    number_of_frames = min(number_of_frames, len(dataset))

    with torch.no_grad():
        for i in tqdm(range(number_of_frames), 'train'):
            if i % 2 == 1:
                continue
            data = dataset[i]

            frame_transform = dataset.transform_manager.get_transform('ld_cc', 'map')
            position = v3d.create(frame_transform[0, 3], frame_transform[1, 3], frame_transform[2, 3])

            if prev_position is not None:
                delta_distance = v3d.length(v3d.sub(position, prev_position))
                if delta_distance < 2.0:
                    continue

            image = data['fc_near']
            pcd = torch.from_numpy(data['ld_cc']).cuda()
            model_input = model.backbone.prepare_input(pcd)

            model_input = KittiDataset.collate_batch([model_input])
            for key, val in model_input.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_input[key] = torch.from_numpy(val).float().cuda()
            batch_dict = model(model_input, metric_head=False, compute_rotation=False, compute_transl=False)
            embedding = batch_dict['out_embedding'][0].cpu().numpy()

            train_positions.append(position)
            train_embeddings.append(embedding)

            cv2.imshow('image', image)
            cv2.waitKey(13)

            prev_position = position

    prev_position: Optional[v3d.Vector3d] = None

    print('Query step..')

    output_distances = []
    index = faiss.IndexFlatL2(256)
    index.add(np.array(train_embeddings))

    with torch.no_grad():
        for i in tqdm(range(number_of_frames), 'query'):
            if i % 2 == 0:
                continue
            data = dataset[i]

            frame_transform = dataset.transform_manager.get_transform('ld_cc', 'map')
            position = v3d.create(frame_transform[0, 3], frame_transform[1, 3], frame_transform[2, 3])

            if prev_position is not None:
                delta_distance = v3d.length(v3d.sub(position, prev_position))
                if delta_distance < 2.0:
                    continue

            image = data['fc_near']
            pcd = torch.from_numpy(data['ld_cc']).cuda()
            model_input = model.backbone.prepare_input(pcd)

            model_input = KittiDataset.collate_batch([model_input])
            for key, val in model_input.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_input[key] = torch.from_numpy(val).float().cuda()
            batch_dict = model(model_input, metric_head=False, compute_rotation=False, compute_transl=False)
            embedding = batch_dict['out_embedding'][0].cpu().numpy()

            d, i = index.search(np.array([embedding]), 1)
            d, i = d[0], i[0]

            train_position = train_positions[int(i[0])]
            distance = v3d.length(v3d.sub(train_position, position))
            output_distances.append(distance)

            cv2.imshow('image', image)
            cv2.waitKey(13)

    df = pd.DataFrame(dict(
        frame=[i for i in range(len(output_distances))],
        distances=output_distances
    ))
    fig = px.line(df, x='frame', y='distances', title='Localication offsets')
    fig.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='', help="Path to dataset")
    parser.add_argument('--dataset_id', default='', help='Dataset id')
    parser.add_argument('--weights_path', default='', required=True, help='Path to model weights')
    args = parser.parse_args()

    if args.dataset_id:
        clearml_dataset = clearml.Dataset.get(dataset_id=args.dataset_id)
        dataset_path = Path(clearml_dataset.get_local_copy())
    elif args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        raise 'Dataset is not set - "dataset_path" or "dataset_id" argument is needed'

    dataset = AstralDatasetReader(dataset_path, 'm11', ['ld_cc', 'fc_near'])

    main(dataset, args.weights_path)
