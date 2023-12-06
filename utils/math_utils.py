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

"""Math utilities for rotations and vector algebra."""

import numpy as np
import torch
import transforms3d


def get_rot6d_from_rot3d(rot3d):
  global_rotation = np.array(
      transforms3d.euler.euler2mat(rot3d[0], rot3d[1], rot3d[2])
  )
  return global_rotation.T.reshape(9)[:6]


def robust_compute_rotation_matrix_from_ortho6d(poses):
  """TODO(jmattarian): Code from: XXXXXXX."""

  x_raw = poses[:, 0:3]
  y_raw = poses[:, 3:6]

  x = normalize_vector(x_raw)
  y = normalize_vector(y_raw)
  middle = normalize_vector(x + y)
  orthmid = normalize_vector(x - y)
  x = normalize_vector(middle + orthmid)
  y = normalize_vector(middle - orthmid)
  z = normalize_vector(cross_product(x, y))

  x = x.view(-1, 3, 1)
  y = y.view(-1, 3, 1)
  z = z.view(-1, 3, 1)
  matrix = torch.cat((x, y, z), 2)
  return matrix


def normalize_vector(v):
  batch = v.shape[0]
  v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
  v_mag = torch.max(v_mag, v.new([1e-8]))
  v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
  v = v / v_mag
  return v


def cross_product(u, v):
  batch = u.shape[0]
  i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
  j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
  k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

  out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

  return out
