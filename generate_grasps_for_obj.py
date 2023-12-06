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

"""Script that performs inference on GeoMatch to produce keypoints for a given object and all end-effectors, followed by the IK used for the paper.

Note: Any other IK could be used.
"""

import argparse
import json
import os
import config
from models.geomatch import GeoMatch
import numpy as np
import plotly.graph_objects as go
import scipy
import torch
from torch import nn
import trimesh as tm
from utils.general_utils import get_handmodel
from utils.gnn_utils import euclidean_min_dist
from utils.gnn_utils import plot_mesh
from utils.gnn_utils import plot_point_cloud


def compute_pose_from_rotation_matrix(t_pose, r_matrix):
  """Computes a 6D pose from a rotation matrix."""

  batch = r_matrix.shape[0]
  joint_num = 2
  r_matrices = (
      r_matrix.view(batch, 1, 3, 3)
      .expand(batch, joint_num, 3, 3)
      .contiguous()
      .view(batch * joint_num, 3, 3)
  )
  src_poses = (
      t_pose.view(1, joint_num, 3, 1)
      .expand(batch, joint_num, 3, 1)
      .contiguous()
      .view(batch * joint_num, 3, 1)
  )

  out_poses = torch.matmul(r_matrices, src_poses.double())

  return out_poses.view(batch, joint_num * 3)


def rotation_matrix_from_vectors(vec1, vec2):
  """Returns the rotation matrix that aligns two vectors.

  Source:
  https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space

  Args:
    vec1: first vector.
    vec2: second vector.

  Returns:
    A 3x3 rotation_matrix/
  """
  a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
      vec2 / np.linalg.norm(vec2)
  ).reshape(3)
  v = np.cross(a, b)
  c = np.dot(a, b)
  s = np.linalg.norm(v)
  kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
  return rotation_matrix


def get_heuristic_init_pose(
    gripper_model, centroids, object_point_cloud, robot_point_cloud, q_rest
):
  """Gets an initial pose based on heuristics, to start IK."""

  _, sorted_vert = euclidean_min_dist(centroids, robot_point_cloud)
  hand_closest_point = robot_point_cloud[sorted_vert[0]]
  center_mass = object_point_cloud.mean(dim=0)

  _, sorted_vert_obj = euclidean_min_dist(centroids, object_point_cloud)
  obj_closest_point = object_point_cloud[sorted_vert_obj[0]]

  hand_vec_normal = hand_closest_point - torch.tensor(centroids)
  obj_vec_normal = obj_closest_point - center_mass

  rot_mat = rotation_matrix_from_vectors(hand_vec_normal, obj_vec_normal)

  new_q_rot = compute_pose_from_rotation_matrix(
      q_rest.squeeze()[3:9], torch.tensor(rot_mat[None])
  )

  new_hand_closest = np.matmul(rot_mat, hand_closest_point)
  trans = obj_closest_point - new_hand_closest

  gripper_model.global_rotation = torch.tensor(rot_mat[None])
  gripper_model.global_translation = trans[None]

  heuristic_q = np.concatenate(
      (trans, new_q_rot.squeeze(), q_rest.squeeze()[9:])
  )
  heuristic_q = torch.tensor(heuristic_q).float()

  return heuristic_q


def autoregressive_inference(
    contact_map_pred,
    match_model,
    top_k,
    obj_pc,
    robot_embed,
    obj_embed,
):
  """Performs the autoregressive inference part of GeoMatch."""

  max_topk = max(top_k)

  with torch.no_grad():
    max_per_kp = torch.topk(contact_map_pred, k=(max_topk + 1), dim=1)
    all_grasps = []

    for k in top_k:
      pred_curr = None
      grasp_points = []
      contact_or_not = []

      obj_proj_embed = match_model.obj_proj(obj_embed)
      robot_proj_embed = match_model.robot_proj(robot_embed)

      for i_prev in range(config.keypoint_n - 1):
        model_kp = match_model.kp_ar_model_1

        if i_prev == 1:
          model_kp = match_model.kp_ar_model_2
        elif i_prev == 2:
          model_kp = match_model.kp_ar_model_3
        elif i_prev == 3:
          model_kp = match_model.kp_ar_model_4
        elif i_prev == 4:
          model_kp = match_model.kp_ar_model_5

        xyz_prev = torch.gather(
            obj_pc[None],
            1,
            max_per_kp.indices[:, k, i_prev, :].repeat(1, 1, 3),
        )

        if i_prev == 0:
          grasp_points.append(xyz_prev.squeeze())
          contact_or_not.append(torch.tensor(1))
        else:
          xyz_prev = torch.stack(grasp_points, dim=0)[None]

        pred_curr = model_kp(
            obj_proj_embed, obj_pc[None], robot_proj_embed, xyz_prev
        )
        pred_prob = nn.Sigmoid()(pred_curr)
        vert_pred = torch.max(pred_prob[..., 0], dim=-1)
        min_idx = vert_pred.indices[0]
        contact_or_not.append(torch.tensor(int(vert_pred.values[0] >= 0.5)))

        pred_curr = obj_pc[min_idx]
        grasp_points.append(pred_curr)

      grasp_points = torch.stack(grasp_points, dim=0)
      contact_or_not = torch.stack(contact_or_not, dim=0)
      final_grasp_points = torch.cat(
          (grasp_points, contact_or_not[..., None]), dim=-1
      )
      all_grasps.append(final_grasp_points)

    return torch.stack(all_grasps, dim=0)


def inference(
    geomatch_model,
    top_k,
    obj_pc,
    obj_adjacency,
    robot_point_cloud,
    robot_adjacency,
    keypoints_idx,
):
  """Performs full inference for GeoMatch."""

  with torch.no_grad():
    obj_embed = geomatch_model.encode_embed(
        geomatch_model.obj_encoder, obj_pc[None], obj_adjacency[None]
    )
    robot_embed = geomatch_model.encode_embed(
        geomatch_model.robot_encoder,
        robot_point_cloud[None],
        robot_adjacency[None],
    )

    robot_feat_size = robot_embed.shape[2]
    keypoint_feat = torch.gather(
        robot_embed,
        1,
        keypoints_idx[..., None].long().repeat(1, 1, robot_feat_size),
    )
    contact_map_pred = torch.matmul(obj_embed, keypoint_feat.transpose(2, 1))[
        ..., None
    ]

    top_obj_contact_kps_pred = autoregressive_inference(
        contact_map_pred,
        geomatch_model,
        top_k,
        obj_pc,
        robot_embed,
        obj_embed,
    )
    pred_points = top_obj_contact_kps_pred

    return pred_points


def inverse_kinematics_optimization(gripper_model, q_init, target_points):
  """Function performing IK optimization and returning a pose."""

  def optimize_target(q):
    q = torch.tensor(q).float()
    source_points, _, _ = gripper_model.get_static_key_points(q.unsqueeze(0))
    e = [
        np.linalg.norm(source_points[i] - target_points[i])
        for i in range(len(source_points))
    ]
    return e

  real_bounds = []

  for _ in range(3):
    real_bounds.append((-0.5, 0.5))

  for _ in range(6):
    real_bounds.append((-np.pi, np.pi))

  for idx, _ in enumerate(gripper_model.revolute_joints_q_lower.squeeze()):
    real_bounds.append((
        gripper_model.revolute_joints_q_lower[:, idx].squeeze().item(),
        gripper_model.revolute_joints_q_upper[:, idx].squeeze().item(),
    ))

  result = scipy.optimize.least_squares(
      optimize_target, q_init, method='trf', bounds=tuple(zip(*real_bounds))
  )
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=0, help='Random seed.')
  parser.add_argument(
      '--device', type=str, default='cpu', help='Use cuda if available'
  )
  parser.add_argument(
      '--object_name',
      type=str,
      default='contactdb+rubber_duck',
      help='Which object to calculate grasp for.',
  )
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/data/grasp_gnn',
      help='Base data directory.',
  )
  parser.add_argument(
      '--saved_model_dir',
      type=str,
      default=(
          'logs_train/exp-pos_weight_500_200_6_kps_final-1683209255.5473607/'
      ),
  )
  parser.add_argument('--output_dir', type=str, default='logs_out_grasps/')
  parser.add_argument('--plot_grasps', default=False, action='store_true')
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  data_dir = args.data_dir
  output_dir = os.path.join(args.output_dir, args.object_name)
  os.makedirs(output_dir, exist_ok=True)

  object_name = args.object_name
  object_mesh_basedir = os.path.join(data_dir, 'object')
  object_mesh_path = os.path.join(
      object_mesh_basedir,
      f'{args.object_name.split("+")[0]}',
      f'{args.object_name.split("+")[1]}',
      f'{args.object_name.split("+")[1]}.stl',
  )
  obj_mesh = tm.load(object_mesh_path)
  obj_normals = obj_mesh.vertex_normals
  plot_grasps = args.plot_grasps

  robot_centroids = json.load(
      open(os.path.join(data_dir, 'robot_centroids.json'))
  )
  robot_list = [
      'ezgripper',
      'barrett',
      'shadowhand',
  ]
  top_ks = [0, 20, 50, 100]

  model = GeoMatch(config)
  model.load_state_dict(
      torch.load(
          os.path.join(args.saved_model_dir, 'weights/grasp_gnn.pth'),
          map_location=torch.device('cpu'),
      )
  )

  model.eval()

  robot_pc_adj = torch.load(
      os.path.join(data_dir, 'gnn_robot_adj_point_clouds_new.pt')
  )

  object_pc_adj = torch.load(
      os.path.join(data_dir, 'gnn_obj_adj_point_clouds_new.pt')
  )
  new_object_pc_adj = torch.load(
      os.path.join(data_dir, 'gnn_obj_adj_point_clouds_new_unseen.pt')
  )
  object_pc_adj.update(new_object_pc_adj)

  object_pc = object_pc_adj[object_name][1]
  obj_adj = object_pc_adj[object_name][0]
  generated_grasps = []

  for robot_name in robot_list:
    hand_model = get_handmodel(robot_name, 1, 'cpu', 1.0)
    robot_pc = robot_pc_adj[robot_name][1]
    robot_adj = robot_pc_adj[robot_name][0]

    robot_keypoints_idx = robot_pc_adj[robot_name][3]
    rest_pose = hand_model.rest_pose
    hand_centroids = robot_centroids[robot_name]

    q_heuristic = get_heuristic_init_pose(
        hand_model, hand_centroids, object_pc, robot_pc, rest_pose
    )

    all_grasps_predicted_keypoints = inference(
        model,
        top_ks,
        object_pc,
        obj_adj,
        robot_pc,
        robot_adj,
        robot_keypoints_idx,
    )

    for i in range(all_grasps_predicted_keypoints.shape[0]):
      predicted_keypoints = all_grasps_predicted_keypoints[i]
      closest_mesh_idxs = [
          euclidean_min_dist(x, obj_mesh.vertices)[1][0]
          for x in predicted_keypoints[:, :3]
      ]
      closest_normals = obj_normals[closest_mesh_idxs]

      pregrasp_pred_keypoints = predicted_keypoints[
          :, :3
      ] + 0.005 * closest_normals.astype('float32')

      res = inverse_kinematics_optimization(
          hand_model, q_heuristic, pregrasp_pred_keypoints
      )
      q_calc = res.x
      q_calc = torch.tensor(q_calc).float()

      calc_key_points, _, _ = hand_model.get_static_key_points(
          q_calc.unsqueeze(0)
      )

      sample = {
          'object_name': object_name,
          'robot_name': robot_name,
          'pred_keypoints': predicted_keypoints,
          'robot_final_keypoints': calc_key_points,
          'init_pose': q_heuristic,
          'pred_grasp_pose': q_calc,
          'sample_idx': i,
          'scale': 1.0,
      }

      if plot_grasps:
        data = [
            plot_mesh(mesh=tm.load(object_mesh_path), opacity=1.0, color='blue')
        ]
        data += hand_model.get_plotly_data(
            q=q_calc.unsqueeze(0), opacity=1.0, color='pink'
        )
        data += [plot_point_cloud(predicted_keypoints, color='green')]
        data += [plot_point_cloud(calc_key_points, color='purple')]

        fig = go.Figure(data=data)
        fig.show()

      generated_grasps.append(sample)

  data_dict = {
      'info': [
          'object_name',
          'robot_name',
          'pred_keypoints',
          'robot_final_keypoints',
          'init_pose',
          'pred_grasp_pose',
          'sample_idx',
      ],
      'metadata': generated_grasps,
  }
  torch.save(data_dict, os.path.join(output_dir, 'gen_grasps.pt'))
