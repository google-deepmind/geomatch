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

"""Training script for GeoMatch."""

import argparse
import dataclasses
import os
import shutil
import sys
import time
import config
import matplotlib.pyplot as plt
from models.geomatch import GeoMatch
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.gnn_utils import square
from utils.gnn_utils import train_metrics
from utils_data.gnn_dataset import GNNDataset


@dataclasses.dataclass
class TrainState:
  geomatch_model: GeoMatch
  train_step: int
  eval_step: int
  writer: SummaryWriter
  optimizer: optim.Optimizer
  epoch: int


def train(global_args, state: TrainState, dataloader: DataLoader):
  """Full training function."""

  state.geomatch_model.train()
  state.optimizer.zero_grad()

  loss_history = []
  acc_history = []

  for _, data in enumerate(
      tqdm(dataloader, desc=f'EPOCH[{state.epoch}/{global_args.epochs}]')
  ):
    state.train_step += 1

    (
        obj_adj,
        obj_features,
        obj_contacts,
        robot_adj,
        robot_features,
        robot_key_point_idx,
        robot_contacts,
        top_obj_contact_kps,
        _,
        _,
        _,
        _,
    ) = data

    if global_args.device == 'cuda':
      obj_features = obj_features.cuda()
      obj_adj = obj_adj.cuda()
      obj_contacts = obj_contacts.cuda()
      robot_adj = robot_adj.cuda()
      robot_features = robot_features.cuda()
      robot_key_point_idx = robot_key_point_idx.cuda().long()
      robot_contacts = robot_contacts.cuda()
      top_obj_contact_kps = top_obj_contact_kps.cuda()

    gt_contact_map = (
        (obj_contacts * robot_contacts.repeat(1, 1, config.obj_pc_n))
        .transpose(2, 1)[..., None]
        .contiguous()
    )

    contact_map_pred, pred_curr = state.geomatch_model(
        obj_features,
        robot_features,
        robot_key_point_idx,
        obj_adj,
        robot_adj,
        top_obj_contact_kps,
    )

    loss_train = state.geomatch_model.calc_loss(
        gt_contact_map,
        contact_map_pred,
        pred_curr,
        gt_contact_map[:, :, 1 : config.keypoint_n, :],
    )
    (
        acc,
        _,
        _,
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
    ) = train_metrics(pred_curr, gt_contact_map[:, :, 1 : config.keypoint_n, :])

    state.optimizer.zero_grad()
    loss_train.backward()

    nn.utils.clip_grad_value_(state.geomatch_model.parameters(), clip_value=1.0)
    state.optimizer.step()

    loss_history.append(loss_train)
    acc_history.append(acc)

    if state.train_step % 10 == 0:
      loss = torch.mean(torch.stack(loss_history))
      square(true_positives, false_positives, true_negatives, false_negatives)
      precision_recall_square = plt.imread('square.jpg').transpose(2, 0, 1)

      state.writer.add_scalar(
          'train/loss', loss.item(), global_step=state.train_step
      )
      state.writer.add_image(
          'train/precision_recall',
          precision_recall_square,
          global_step=state.train_step,
      )

      plt.close()

  epoch_loss = torch.mean(torch.stack(loss_history))
  epoch_accuracy = torch.mean(torch.stack(acc_history))

  if state.epoch % 1 == 0:
    print(
        f'[train] loss on {state.epoch}: {epoch_loss}\n'
        f'           accuracy: {epoch_accuracy}\n'
    )


def validate(global_args, state: TrainState, dataloader: DataLoader):
  """Full evaluation function."""

  with torch.no_grad():
    state.geomatch_model.eval()

    loss_history = []
    acc_history = []

    for data in tqdm(
        dataloader, desc=f'EPOCH[{state.epoch}/{global_args.epochs}]'
    ):
      state.eval_step += 1

      (
          obj_adj,
          obj_features,
          obj_contacts,
          robot_adj,
          robot_features,
          robot_key_point_idx,
          robot_contacts,
          top_obj_contact_kps,
          _,
          _,
          _,
          _,
      ) = data

      if global_args.device == 'cuda':
        obj_features = obj_features.cuda()
        obj_adj = obj_adj.cuda()
        obj_contacts = obj_contacts.cuda()
        robot_adj = robot_adj.cuda()
        robot_features = robot_features.cuda()
        robot_key_point_idx = robot_key_point_idx.cuda().long()
        robot_contacts = robot_contacts.cuda()
        top_obj_contact_kps = top_obj_contact_kps.cuda()

      gt_contact_map = (
          (obj_contacts * robot_contacts.repeat(1, 1, config.obj_pc_n))
          .transpose(2, 1)[..., None]
          .contiguous()
      )

      contact_map_pred, pred_curr = state.geomatch_model(
          obj_features,
          robot_features,
          robot_key_point_idx,
          obj_adj,
          robot_adj,
          top_obj_contact_kps,
      )

      loss_val = state.geomatch_model.calc_loss(
          gt_contact_map,
          contact_map_pred,
          pred_curr,
          gt_contact_map[:, :, 1 : config.keypoint_n, :],
      )
      (
          acc,
          _,
          _,
          true_positives,
          true_negatives,
          false_positives,
          false_negatives,
      ) = train_metrics(
          pred_curr, gt_contact_map[:, :, 1 : config.keypoint_n, :]
      )

      loss_history.append(loss_val)
      acc_history.append(acc)

      if state.eval_step % 10 == 0:
        loss = torch.mean(torch.stack(loss_history))
        square(true_positives, false_positives, true_negatives, false_negatives)
        precision_recall_square = plt.imread('square.jpg').transpose(2, 0, 1)

        state.writer.add_scalar(
            'validate/loss', loss.item(), global_step=state.eval_step
        )
        state.writer.add_image(
            'validate/precision_recall',
            precision_recall_square,
            global_step=state.eval_step,
        )

        plt.close()

    epoch_loss = torch.mean(torch.stack(loss_history))
    epoch_accuracy = torch.mean(torch.stack(acc_history))
    print(
        f'[validate] loss: {epoch_loss}\n'
        f'           accuracy: {epoch_accuracy}\n'
    )


if __name__ == '__main__':
  start_time = time.time()
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--exp_name', type=str, default='all_end_effectors_object_split'
  )
  parser.add_argument('--seed', type=int, default=42, help='Random seed.')
  parser.add_argument('--batch_size', type=int, default=64, help='Random seed.')
  parser.add_argument(
      '--device', type=str, default='cuda', help='Use cuda if available'
  )
  parser.add_argument(
      '--epochs', type=int, default=200, help='Number of epochs to train.'
  )
  parser.add_argument(
      '--lr', type=float, default=1e-4, help='Initial learning rate.'
  )
  parser.add_argument(
      '--weight_decay',
      type=float,
      default=0.0,
      help='Weight decay (L2 loss on parameters).',
  )
  parser.add_argument(
      '--out_features',
      type=int,
      default=512,
      help='Number of object and end-effector feature dimension.',
  )
  parser.add_argument(
      '--robot_weighting',
      type=int,
      default=500,
      help='Weight for full distribution BCE class loss.',
  )
  parser.add_argument(
      '--matchnet_weighting',
      type=int,
      default=200,
      help='Weight for matching BCE class loss.',
  )
  parser.add_argument(
      '--exclude_ee_list',
      nargs='+',
      help='End-effector to be excluded from the default list.',
  )

  parser.add_argument(
      '--dataset_basedir',
      type=str,
      default='/data/grasp_gnn',
      help='Path to data.',
  )

  args = parser.parse_args()
  args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  if args.device == 'cuda':
    torch.cuda.manual_seed_all(args.seed)

  config.obj_out_feats = args.out_features
  config.robot_out_feats = config.obj_out_feats
  config.robot_weighting = args.robot_weighting
  config.matchnet_weighting = args.matchnet_weighting

  robot_name_list = [
      'ezgripper',
      'barrett',
      'robotiq_3finger',
      'allegro',
      'shadowhand',
  ]

  if args.exclude_ee_list is not None:
    for ee in args.exclude_ee_list:
      robot_name_list.remove(ee)

  log_dir = os.path.join('logs_train', f'exp-{args.exp_name}-{str(start_time)}')
  weight_dir = os.path.join(log_dir, 'weights')
  tb_dir = os.path.join(log_dir, 'tb_dir')
  shutil.rmtree(log_dir, ignore_errors=True)
  os.makedirs(log_dir, exist_ok=True)
  os.makedirs(weight_dir, exist_ok=True)
  os.makedirs(tb_dir, exist_ok=True)
  f = open(os.path.join(log_dir, 'command.txt'), 'w')
  f.write(' '.join(sys.argv))
  f.close()
  writer = SummaryWriter(log_dir=tb_dir)

  dataset_basedir = args.dataset_basedir

  batchsize = args.batch_size
  train_dataset = GNNDataset(
      dataset_basedir=dataset_basedir,
      mode='train',
      device=args.device,
      robot_name_list=robot_name_list,
  )
  train_dataloader = DataLoader(
      dataset=train_dataset,
      batch_size=batchsize,
      shuffle=True,
      num_workers=0,
  )

  validate_dataset = GNNDataset(
      dataset_basedir=dataset_basedir,
      mode='validate',
      device=args.device,
      robot_name_list=robot_name_list,
  )
  validate_dataloader = DataLoader(
      dataset=validate_dataset,
      batch_size=batchsize,
      shuffle=True,
      num_workers=0,
  )

  geomatch_model = GeoMatch(config)
  geomatch_model = geomatch_model.to(args.device)

  optimizer = optim.Adam(
      list(geomatch_model.parameters()),
      lr=args.lr,
      weight_decay=args.weight_decay,
      betas=(0.9, 0.99),
  )

  torch.save(
      geomatch_model.state_dict(), os.path.join(weight_dir, 'grasp_gnn.pth')
  )

  train_step = 0
  val_step = 0

  train_state = TrainState(
      geomatch_model=geomatch_model,
      writer=writer,
      optimizer=optimizer,
      train_step=train_step,
      eval_step=val_step,
      epoch=0,
  )

  for i_epoch in range(args.epochs):
    train(args, train_state, train_dataloader)
    validate(args, train_state, validate_dataloader)

  torch.save(
      geomatch_model.state_dict(), os.path.join(weight_dir, 'grasp_gnn.pth')
  )
  writer.close()
  print(f'consuming time: {time.time() - start_time}')
