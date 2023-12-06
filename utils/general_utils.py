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

"""General utilities."""

import json
import os
import random
import numpy as np
import torch
from utils import gripper_utils


def get_handmodel(
    robot, batch_size, device, hand_scale=1.0, data_dir='/data/grasp_gnn'
):
  """Fetches the hand model object for a given gripper."""
  urdf_assets_meta = json.load(
      open(os.path.join(data_dir, 'urdf/urdf_assets_meta.json'))
  )
  urdf_path = urdf_assets_meta['urdf_path'][robot].replace('data', data_dir)
  meshes_path = urdf_assets_meta['meshes_path'][robot].replace('data', data_dir)
  hand_model = gripper_utils.HandModel(
      robot,
      urdf_path,
      meshes_path,
      batch_size=batch_size,
      device=device,
      hand_scale=hand_scale,
      data_dir=data_dir,
  )
  return hand_model


def set_global_seed(seed=42):
  torch.cuda.manual_seed_all(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
