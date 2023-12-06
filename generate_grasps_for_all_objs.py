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

"""Batch script that generates grasps for all eval objects and all end-effectors."""

import argparse
import json
import os
import subprocess
import numpy as np
import torch


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=0, help='Random seed.')
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

  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  data_dir = args.data_dir
  print(f'Saved model dir: {args.saved_model_dir}')

  object_list = json.load(
      open(
          os.path.join(
              data_dir,
              'CMapDataset-sqrt_align/split_train_validate_objects.json',
          ),
          'rb',
      )
  )['validate']

  ps = [
      subprocess.Popen(
          [
              'python',
              'generate_grasps_for_obj.py',
              '--object_name',
              object_name,
              '--data_dir',
              data_dir,
              '--saved_model_dir',
              args.saved_model_dir,
              '--output_dir',
              args.output_dir,
          ],
          stdout=subprocess.PIPE,
      )
      for object_name in object_list
  ]

  exit_codes = [p.wait() for p in ps]

  finished_proc_inds = [i for i, p in enumerate(ps) if p.poll() is not None]

  print(f'Exit codes of all processes: {exit_codes}')
  print(f'All processes finished?: {finished_proc_inds}')
