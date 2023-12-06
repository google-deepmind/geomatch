# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements Pytorch dataloader class needed for GeoMatch."""

import json
import os
import torch
from torch.utils import data
from utils_data import augmentors


class GNNDataset(data.Dataset):
  """Dataloader class for GeoMatch."""

  def __init__(
      self,
      dataset_basedir,
      object_npts=2048,
      device='cuda' if torch.cuda.is_available() else 'cpu',
      mode='train',
      robot_name_list=None,
  ):
    self.device = device
    self.dataset_basedir = dataset_basedir
    self.object_npts = object_npts

    if not robot_name_list:
      self.robot_name_list = [
          'ezgripper',
          'barrett',
          'robotiq_3finger',
          'allegro',
          'shadowhand',
      ]
    else:
      self.robot_name_list = robot_name_list

    print('loading object point clouds and adjacency matrices....')
    self.object_pc_adj = torch.load(
        os.path.join(dataset_basedir, 'gnn_obj_adj_point_clouds_new.pt')
    )

    print('loading robot point clouds and adjacency matrices...')
    self.robot_pc_adj = torch.load(
        os.path.join(dataset_basedir, 'gnn_robot_adj_point_clouds_new.pt')
    )

    print('loading object/robot cmaps....')
    cmap_dataset = torch.load(
        os.path.join(dataset_basedir, 'gnn_obj_cmap_robot_cmap_adj_new.pt')
    )['metadata']

    self.metadata = cmap_dataset

    if mode == 'train':
      self.object_list = json.load(
          open(
              os.path.join(
                  dataset_basedir,
                  'CMapDataset-sqrt_align/split_train_validate_objects.json',
              ),
              'rb',
          )
      )[mode]
      self.metadata = [
          t
          for t in self.metadata
          if t[6] in self.object_list and t[7] in self.robot_name_list
      ]
    elif mode == 'validate':
      self.object_list = json.load(
          open(
              os.path.join(
                  dataset_basedir,
                  'CMapDataset-sqrt_align/split_train_validate_objects.json',
              ),
              'rb',
          )
      )[mode]
      self.metadata = [
          t
          for t in self.metadata
          if t[6] in self.object_list and t[7] in self.robot_name_list
      ]
    elif mode == 'full':
      self.object_list = (
          json.load(
              open(
                  os.path.join(
                      dataset_basedir,
                      'CMapDataset-sqrt_align/split_train_validate_objects.json',
                  ),
                  'rb',
              )
          )['train']
          + json.load(
              open(
                  os.path.join(
                      dataset_basedir, 'split_train_validate_objects.json'
                  ),
                  'rb',
              )
          )['validate']
      )
      self.metadata = [
          t
          for t in self.metadata
          if t[6] in self.object_list and t[7] in self.robot_name_list
      ]
    else:
      raise NotImplementedError()
    print(f'object selection: {self.object_list}')

    self.mode = mode

    self.datasize = len(self.metadata)
    print('finish loading dataset....')

    self.rotate = augmentors.PointcloudRotatePerturbation(
        angle_sigma=0.03, angle_clip=0.1
    )
    self.translate = augmentors.PointcloudTranslate(translate_range=0.01)
    self.jitter = augmentors.PointcloudJitter(std=0.04, clip=0.1)

  def __len__(self):
    return self.datasize

  def __getitem__(self, item):
    object_name = self.metadata[item][6]
    robot_name = self.metadata[item][7]

    obj_adj = self.object_pc_adj[object_name][0]
    obj_contacts = self.metadata[item][0]
    obj_features = self.object_pc_adj[object_name][1]

    if self.mode in ['train']:
      obj_features = self.rotate(obj_features)

    robot_adj = self.robot_pc_adj[robot_name][0]
    robot_features = self.robot_pc_adj[robot_name][1]
    robot_key_point_idx = self.robot_pc_adj[robot_name][3]
    assert robot_key_point_idx.shape[0] == 6
    robot_contacts = self.metadata[item][1]
    top_obj_contact_kps = self.metadata[item][2]
    assert top_obj_contact_kps.shape[0] == 6
    top_obj_contact_verts = self.metadata[item][3]
    assert top_obj_contact_verts.shape[0] == 6
    full_obj_contact_map = self.metadata[item][4]

    if self.mode in ['train']:
      robot_features = self.rotate(robot_features)

    return (
        obj_adj,
        obj_features,
        obj_contacts,
        robot_adj,
        robot_features,
        robot_key_point_idx,
        robot_contacts,
        top_obj_contact_kps,
        top_obj_contact_verts,
        full_obj_contact_map,
        object_name,
        robot_name,
    )
