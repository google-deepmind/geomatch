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

"""Script that preprocesses the data into the format expected by GeoMatch."""

import argparse
import os
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import trimesh as tm
from utils.general_utils import get_handmodel
from utils.gnn_utils import euclidean_min_dist
from utils.gnn_utils import generate_adj_mat_feats
from utils.gnn_utils import generate_contact_maps

device = 'cpu'

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_basedir', type=str, default='/data/grasp_gnn')
  parser.add_argument(
      '--object_mesh_basedir', type=str, default='/data/grasp_gnn/object'
  )
  parser.add_argument('--gnn_dataset_basedir', type=str, default='data')

  args = parser.parse_args()

  cmap_data = torch.load(
      os.path.join(
          args.dataset_basedir, 'CMapDataset-sqrt_align/cmap_dataset.pt'
      ),
      map_location=torch.device('cpu'),
  )
  object_data = torch.load(
      os.path.join(
          args.dataset_basedir, 'CMapDataset-sqrt_align/object_point_clouds.pt'
      )
  )

  robot_name_list = [
      'ezgripper',
      'barrett',
      'robotiq_3finger',
      'allegro',
      'shadowhand',
  ]
  hand_model = {}
  robot_key_point_idx = {}
  surface_points_per_robot = {}
  threshold = 0.04

  data_dict = {}
  for robot_name in tqdm(robot_name_list):
    hand_model[robot_name] = get_handmodel(
        robot_name, 1, 'cpu', 1.0, data_dir=args.dataset_basedir
    )
    joint_lower = np.array(
        hand_model[robot_name].revolute_joints_q_lower.cpu().reshape(-1)
    )
    joint_upper = np.array(
        hand_model[robot_name].revolute_joints_q_upper.cpu().reshape(-1)
    )
    joint_mid = (joint_lower + joint_upper) / 2
    joints_q = (joint_mid + joint_lower) / 2
    rest_pose = (
        torch.from_numpy(
            np.concatenate([np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]), joints_q])
        )
        .unsqueeze(0)
        .to(device)
        .float()
    )
    surface_points = (
        hand_model[robot_name]
        .get_surface_points(rest_pose, downsample=True)
        .cpu()
        .squeeze(0)
    )
    key_points, key_point_idx_dict, surface_sample_kp_idx = hand_model[
        robot_name
    ].get_static_key_points(rest_pose, surface_points)
    robot_key_point_idx[robot_name] = key_point_idx_dict
    surface_points_per_robot[robot_name] = surface_points
    robot_adj, robot_features = generate_adj_mat_feats(surface_points, knn=8)
    data_dict[robot_name] = (
        robot_adj,
        robot_features,
        rest_pose,
        surface_sample_kp_idx,
        key_point_idx_dict,
        robot_name,
    )

  torch.save(
      data_dict,
      os.path.join(
          args.gnn_dataset_basedir, 'gnn_robot_adj_point_clouds_new.pt'
      ),
  )

  data_dict = {}
  for obj_name in tqdm(object_data):
    object_mesh_path = os.path.join(
        args.object_mesh_basedir,
        f'{obj_name.split("+")[0]}',
        f'{obj_name.split("+")[1]}',
        f'{obj_name.split("+")[1]}.stl',
    )
    obj_point_cloud = object_data[obj_name]
    obj_mesh = tm.load(object_mesh_path)
    normals = []

    for p in obj_point_cloud:
      dist, indices = euclidean_min_dist(p, obj_mesh.vertices)
      normals.append(obj_mesh.vertex_normals[indices[0]])

    normals = np.stack(normals, axis=0)

    obj_adj, obj_features = generate_adj_mat_feats(obj_point_cloud, knn=8)
    data_dict[obj_name] = (obj_adj, obj_features, torch.tensor(normals))

  torch.save(
      data_dict,
      os.path.join(args.gnn_dataset_basedir, 'gnn_obj_adj_point_clouds_new.pt'),
  )

  data_list = []
  for metadata in tqdm(cmap_data['metadata']):
    _, q, object_name, robot_name = metadata
    q = q.unsqueeze(0)
    obj_point_cloud = object_data[object_name]

    robot_grasp_kps, _, _ = hand_model[robot_name].get_static_key_points(
        q, surface_points_per_robot[robot_name]
    )
    obj_contact_map = np.zeros((6, obj_point_cloud.shape[0]))
    full_obj_contact_map = np.zeros((obj_point_cloud.shape[0], 1))

    point_dists_idxs = [
        euclidean_min_dist(x, obj_point_cloud) for x in robot_grasp_kps
    ]
    robot_contact_map = np.array(
        [int(x[0] < threshold) for x in point_dists_idxs]
    ).reshape(-1, 1)
    top_obj_contact_kps = torch.stack(
        [obj_point_cloud[x[1][0]] for x in point_dists_idxs], dim=0
    )
    top_obj_contact_verts = torch.tensor(
        [x[1][0] for x in point_dists_idxs]
    ).long()

    for i in range(top_obj_contact_verts.shape[0]):
      obj_contact_map[i, point_dists_idxs[i][1][:20]] = 1
      full_obj_contact_map[point_dists_idxs[i][1][:20]] = 1

    obj_contacts = generate_contact_maps(obj_contact_map)
    robot_contacts = generate_contact_maps(robot_contact_map)

    data_point = (
        obj_contacts,
        robot_contacts,
        top_obj_contact_kps,
        top_obj_contact_verts,
        full_obj_contact_map,
        q.squeeze(0),
        object_name,
        robot_name,
    )
    data_list.append(data_point)

  data_dict = {
      'info': [
          'obj_contacts',
          'robot_contacts',
          'top_obj_contact_kps',
          'top_obj_contact_verts',
          'full_obj_contact_map',
          'q',
          'object_name',
          'robot_name',
      ],
      'metadata': data_list,
  }
  torch.save(
      data_dict,
      os.path.join(
          args.gnn_dataset_basedir, 'gnn_obj_cmap_robot_cmap_adj_new.pt'
      ),
  )
