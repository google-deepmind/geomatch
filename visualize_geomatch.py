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

"""Visualization helped for GeoMatch predictions."""

import argparse
import itertools
import json
import os
import random
import config
from models.geomatch import GeoMatch
from plotly.subplots import make_subplots
import torch
from torch import nn
from utils.general_utils import get_handmodel
from utils.gnn_utils import plot_point_cloud


def return_random_obj_ee_pair(
    obj_name, rbt_name, obj_data, rbt_data, contact_map_data_gt
):
  """Returns a random object-gripper pair to use for visualization."""

  obj_adj = obj_data[obj_name][0]
  obj_pc = obj_data[obj_name][1]
  robot_adj = rbt_data[rbt_name][0]
  robot_pc = rbt_data[rbt_name][1]
  rest_pose = rbt_data[rbt_name][2]
  keypoints_idx = rbt_data[rbt_name][3]
  keypoints_idx_dict = rbt_data[rbt_name][4]

  found = False
  idx_list = []
  for i, data in enumerate(contact_map_data_gt['metadata']):
    if data[6] == obj_name and data[7] == rbt_name:
      idx_list.append(i)
      found = True
      break

  if not found:
    raise ModuleNotFoundError('Did not find a matching combination, try again!')

  rand_idx = random.choice(idx_list)
  obj_cmap = contact_map_data_gt['metadata'][rand_idx][0]
  robot_cmap = contact_map_data_gt['metadata'][rand_idx][1]
  top_obj_contact_kps = contact_map_data_gt['metadata'][rand_idx][2]
  top_obj_contact_verts = contact_map_data_gt['metadata'][rand_idx][3]
  q = contact_map_data_gt['metadata'][rand_idx][4]

  return (
      obj_adj,
      obj_pc,
      robot_adj,
      robot_pc,
      rest_pose,
      keypoints_idx,
      obj_cmap,
      robot_cmap,
      top_obj_contact_kps,
      top_obj_contact_verts,
      q,
      keypoints_idx_dict,
  )


def autoregressive_inference(
    contact_map_pred, match_model, obj_pc, robot_embed, obj_embed, top_k=0
):
  """Performs the autoregressive inference of GeoMatch."""

  with torch.no_grad():
    max_per_kp = torch.topk(contact_map_pred, k=3, dim=1)
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
          max_per_kp.indices[:, top_k, i_prev, :].repeat(1, 1, 3),
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

      # Projected on object
      pred_curr = obj_pc[min_idx]
      grasp_points.append(pred_curr)

    grasp_points = torch.stack(grasp_points, dim=0)
    contact_or_not = torch.stack(contact_or_not, dim=0)

    return torch.cat((grasp_points, contact_or_not[..., None]), dim=-1)


def plot_side_by_side(
    point_cloud,
    contact_map,
    hand_data,
    i_keypoint,
    save_dir,
    save_plot,
    gt_contact_map=None,
    pred_points=None,
    top_obj_contact_kps=None,
):
  """Side-by-side plots of the object point cloud with the predicted keypoints, gripper with the canonical keypoints and a GT sample for comparison.

  Each keypoint will generate a new plot. The current keypoint is depicted with
  a different color.

  Args:
    point_cloud: the object point cloud
    contact_map: the predicted contact map
    hand_data: gripper data to plot - mesh, point cloud etc.
    i_keypoint: i-th keypoint to plot data for
    save_dir: directory to save plots in
    save_plot: bool, whether to save plots
    gt_contact_map: ground truth contact map for comparison
    pred_points: predicted keypoints to plot
    top_obj_contact_kps: grouth truth contact points to plot
  """
  fig = make_subplots(
      rows=1,
      cols=3,
      specs=[
          [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]
      ],
  )
  fig.add_trace(
      plot_point_cloud(point_cloud, contact_map.squeeze()), row=1, col=1
  )

  if pred_points is not None:
    pred_points = pred_points.detach().numpy()

    for i in range(pred_points.shape[0]):
      c = 'black'
      if pred_points[i, 3] == 1.0:
        c = 'red'
      fig.add_trace(
          plot_point_cloud(pred_points[i, :3][None], color=c, size=5),
          row=1,
          col=1,
      )
    fig.add_trace(
        plot_point_cloud(
            pred_points[i_keypoint][None], color='magenta', size=5
        ),
        row=1,
        col=1,
    )

  for d in hand_data:
    fig.add_trace(d, row=1, col=2)

  if gt_contact_map is not None:
    fig.add_trace(
        plot_point_cloud(point_cloud, gt_contact_map.squeeze()), row=1, col=3
    )

    if top_obj_contact_kps is not None:
      fig.add_trace(
          plot_point_cloud(
              top_obj_contact_kps[i_keypoint, :][None], color='red'
          ),
          row=1,
          col=3,
      )

  fig.update_layout(
      height=800,
      width=1800,
      title_text=(
          f'Prediction on keypoint {i_keypoint} and one GT grasp for'
          ' comparison.'
      ),
  )
  if not save_plot:
    fig.show()
  else:
    fig.write_image(
        os.path.join(save_dir, f'prediction+rand_gt_keypoint_{i_keypoint}'),
        'jpg',
        scale=2,
    )


def plot_predicted_keypoints(
    obj_name,
    rbt_name,
    obj_data,
    rbt_data,
    contact_map_data_gt,
    geomatch_model,
    save_dir,
    save_plot,
    data_basedir,
    top_k=0,
):
  """Generates plots for a predicted grasp for a given object-gripper pair."""
  (
      obj_adj,
      obj_pc,
      robot_adj,
      robot_pc,
      rest_pose,
      keypoints_idx,
      obj_cmap,
      robot_cmap,
      top_obj_contact_kps,
      _,
      _,
      _,
  ) = return_random_obj_ee_pair(
      obj_name, rbt_name, obj_data, rbt_data, contact_map_data_gt
  )

  with torch.no_grad():
    obj_embed = geomatch_model.encode_embed(
        geomatch_model.obj_encoder, obj_pc[None], obj_adj[None]
    )
    robot_embed = geomatch_model.encode_embed(
        geomatch_model.robot_encoder, robot_pc[None], robot_adj[None]
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
    gt_contact_map = (
        (obj_cmap * robot_cmap.repeat(1, config.obj_pc_n))
        .transpose(1, 0)[..., None]
        .contiguous()
    )

    top_obj_contact_kps_pred = autoregressive_inference(
        contact_map_pred, model, obj_pc, robot_embed, obj_embed, top_k
    )
    pred_points = top_obj_contact_kps_pred

  hand_model = get_handmodel(rbt_name, 1, 'cpu', 1.0, data_dir=data_basedir)
  print('PREDICTION: ', pred_points)

  for i in range(contact_map_pred.shape[2]):
    gt_contact_map_i = gt_contact_map[:, i, :]

    obj_kp_cmap = contact_map_pred[:, :, i, :]
    obj_kp_cmap_labels = torch.nn.Sigmoid()(obj_kp_cmap)

    selected_kp = robot_pc[keypoints_idx[i].long(), :][None]
    vis_data = hand_model.get_plotly_data(q=rest_pose, opacity=0.5)
    vis_data += [
        plot_point_cloud(robot_pc[keypoints_idx.long(), :].cpu(), color='black')
    ]
    vis_data += [plot_point_cloud(selected_kp.cpu(), color='red')]

    plot_side_by_side(
        obj_pc,
        obj_kp_cmap_labels.detach().numpy(),
        vis_data,
        i,
        save_dir,
        save_plot,
        gt_contact_map_i,
        pred_points,
        top_obj_contact_kps,
    )


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--object_name', type=str, default='')
  parser.add_argument('--robot_name', type=str, default='')
  parser.add_argument('--random_example', default=True, action='store_true')
  parser.add_argument('--dataset_dir', type=str, default='/data/grasp_gnn')
  parser.add_argument('--save_plots', default=False, action='store_true')
  parser.add_argument('--top_k_idx', type=int, default=0)
  parser.add_argument(
      '--saved_model_dir',
      type=str,
      default='logs_train/exp-pos_weight_500_200_6_kps-1683055568.7644374/',
  )
  args = parser.parse_args()

  dataset_basedir = args.dataset_dir
  saved_model_dir = args.saved_model_dir
  top_k_idx = args.top_k_idx

  saved_model_dir = args.saved_model_dir
  save_plot_dir = os.path.join(saved_model_dir, 'plots')

  if not os.path.exists(save_plot_dir):
    os.mkdir(save_plot_dir)

  device = 'cpu'
  object_data = torch.load(
      os.path.join(dataset_basedir, 'gnn_obj_adj_point_clouds_new.pt')
  )
  robot_data = torch.load(
      os.path.join(dataset_basedir, 'gnn_robot_adj_point_clouds_new.pt')
  )
  cmap_data_gt = torch.load(
      os.path.join(dataset_basedir, 'gnn_obj_cmap_robot_cmap_adj_new.pt'),
      map_location=torch.device('cpu'),
  )

  eval_object_list = json.load(
      open(
          os.path.join(
              dataset_basedir,
              'CMapDataset-sqrt_align/split_train_validate_objects.json',
          ),
          'rb',
      )
  )['validate']
  robot_name_list = [
      'ezgripper',
      'barrett',
      'robotiq_3finger',
      'allegro',
      'shadowhand',
  ]

  obj_robot_pairs = list(itertools.product(eval_object_list, robot_name_list))

  model = GeoMatch(config)
  model.load_state_dict(
      torch.load(
          os.path.join(saved_model_dir, 'weights/grasp_gnn.pth'),
          map_location=torch.device('cpu'),
      )
  )

  model.eval()

  object_name = args.object_name
  robot_name = args.robot_name

  plot_predicted_keypoints(
      object_name,
      robot_name,
      object_data,
      robot_data,
      cmap_data_gt,
      model,
      save_plot_dir,
      args.save_plots,
      dataset_basedir,
      top_k_idx,
  )
