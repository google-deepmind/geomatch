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

"""GeoMatch model definition."""

from models.gnn import GCN
from models.mlp import MLP
import torch
from torch import nn


class GeoMatchARModule(nn.Module):
  """Autoregressive module class for GeoMatch."""

  def __init__(self, config, n_kp) -> None:
    super().__init__()

    self.config = config
    self.n_kp = n_kp
    self.final_fc = MLP(128 + 3 * self.n_kp, 1, 3, 256)

  def forward(self, obj_proj_embed, obj_pc, robot_proj_embed, xyz_prev):
    robot_i_embed = (
        robot_proj_embed[:, self.n_kp][..., None]
        .transpose(2, 1)
        .repeat(1, self.config.obj_pc_n, 1)
    )
    obj_robot_embed = torch.cat((obj_proj_embed, robot_i_embed), dim=-1)

    diff_xyz_tensor = []
    for i in range(self.n_kp):
      diff_xyz = obj_pc - xyz_prev[:, i, :][..., None].transpose(2, 1)
      diff_xyz_tensor.append(diff_xyz)

    diff_xyz_tensor = torch.stack(diff_xyz_tensor, dim=-1)
    diff_xyz_tensor = diff_xyz_tensor.view(
        diff_xyz_tensor.shape[0], diff_xyz_tensor.shape[1], -1
    )
    inp = torch.cat((obj_robot_embed, diff_xyz_tensor), dim=-1)
    pred_curr = self.final_fc(inp)

    return pred_curr

  def calc_loss(self, pred, label):
    pred = pred.view(pred.shape[0] * pred.shape[1], 1)
    label = label.view(label.shape[0] * label.shape[1], 1)

    pos_weight = torch.tensor([1000.0]).cuda()
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred, label)
    return torch.mean(loss)


class GeoMatch(nn.Module):
  """GeoMatch model class."""

  def __init__(self, config) -> None:
    super().__init__()

    self.config = config
    self.n_kp = config.keypoint_n
    self.robot_weighting = config.robot_weighting
    self.match_weighting = config.matchnet_weighting
    self.dist_loss_weight = config.dist_loss_weight
    self.match_loss_weight = config.match_loss_weight

    self.obj_encoder = GCN(
        nfeat=config.obj_in_feats,
        nhid=config.hidden_n,
        nout=config.obj_out_feats,
        dropout=0.5,
        num_hidden=config.num_hidden,
    )

    self.robot_encoder = GCN(
        nfeat=config.robot_in_feats,
        nhid=config.hidden_n,
        nout=config.robot_out_feats,
        dropout=0.5,
        num_hidden=config.num_hidden,
    )

    self.obj_proj = nn.Linear(self.config.obj_out_feats, 64, bias=False)
    self.robot_proj = nn.Linear(self.config.robot_out_feats, 64, bias=False)
    self.kp_ar_model_1 = GeoMatchARModule(config, 1)
    self.kp_ar_model_2 = GeoMatchARModule(config, 2)
    self.kp_ar_model_3 = GeoMatchARModule(config, 3)
    self.kp_ar_model_4 = GeoMatchARModule(config, 4)
    self.kp_ar_model_5 = GeoMatchARModule(config, 5)

  def encode_embed(self, encoder, feature, adj_mat, normalize_emb=True):
    x = encoder(feature, adj_mat)
    if normalize_emb:
      x = x.clone() / torch.norm(x, dim=-1, keepdim=True)
    return x

  def forward(
      self, obj_pc, robot_pc, robot_key_point_idx, obj_adj, robot_adj, xyz_prev
  ):
    obj_embed = self.encode_embed(self.obj_encoder, obj_pc, obj_adj)
    robot_embed = self.encode_embed(self.robot_encoder, robot_pc, robot_adj)

    robot_feat_size = robot_embed.shape[2]
    keypoint_feat = torch.gather(
        robot_embed,
        1,
        robot_key_point_idx[..., None].long().repeat(1, 1, robot_feat_size),
    )
    contact_map_pred = torch.matmul(obj_embed, keypoint_feat.transpose(2, 1))[
        ..., None
    ]

    obj_proj_embed = self.obj_proj(obj_embed)
    robot_proj_embed = self.robot_proj(robot_embed)

    output_1 = self.kp_ar_model_1(
        obj_proj_embed, obj_pc, robot_proj_embed, xyz_prev
    )
    output_2 = self.kp_ar_model_2(
        obj_proj_embed, obj_pc, robot_proj_embed, xyz_prev
    )
    output_3 = self.kp_ar_model_3(
        obj_proj_embed, obj_pc, robot_proj_embed, xyz_prev
    )
    output_4 = self.kp_ar_model_4(
        obj_proj_embed, obj_pc, robot_proj_embed, xyz_prev
    )
    output_5 = self.kp_ar_model_5(
        obj_proj_embed, obj_pc, robot_proj_embed, xyz_prev
    )

    output = torch.cat(
        (output_1, output_2, output_3, output_4, output_5), dim=-1
    )[..., None]

    return contact_map_pred, output

  def calc_loss(self, gt_contact_map, contact_map_pred, pred, label):
    flat_contact_map_pred = contact_map_pred.view(
        contact_map_pred.shape[0]
        * contact_map_pred.shape[1]
        * contact_map_pred.shape[2],
        1,
    )
    flat_gt_contact_map = gt_contact_map.view(
        gt_contact_map.shape[0]
        * gt_contact_map.shape[1]
        * gt_contact_map.shape[2],
        1,
    )

    pos_weight = torch.Tensor([self.robot_weighting]).cuda()
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
        flat_contact_map_pred, flat_gt_contact_map
    )
    l_dist = torch.mean(loss)

    pos_weight = torch.tensor([self.match_weighting]).cuda()

    loss = []
    for i in range(self.n_kp - 1):
      pred_i = pred[:, :, i]
      label_i = label[:, :, i]
      pred_i = pred_i.view(pred_i.shape[0] * pred_i.shape[1], 1)
      label_i = label_i.view(label_i.shape[0] * label_i.shape[1], 1)
      loss.append(nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred_i, label_i))

    loss = torch.stack(loss)
    l_match = torch.mean(loss)

    return self.dist_loss_weight * l_dist + self.match_loss_weight * l_match
