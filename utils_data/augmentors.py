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

"""Augmentation classes for point clouds."""

import numpy as np
import torch


def angle_axis(angle: float, axis: np.ndarray):
  """Returns a 4x4 rotation matrix that performs a rotation around axis by angle."""

  u = axis / np.linalg.norm(axis)
  cosval, sinval = np.cos(angle), np.sin(angle)

  cross_prod_mat = np.array(
      [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]]
  )

  r = torch.from_numpy(
      cosval * np.eye(3)
      + sinval * cross_prod_mat
      + (1.0 - cosval) * np.outer(u, u)
  )
  return r.float()


class PointcloudJitter(object):
  """Adds jitter with a given std to a point cloud."""

  def __init__(self, std=0.005, clip=0.025):
    self.std, self.clip = std, clip

  def __call__(self, points):
    jittered_data = (
        points.new(points.size(0), 3)
        .normal_(mean=0.0, std=self.std)
        .clamp_(-self.clip, self.clip)
    )
    points[:, 0:3] += jittered_data
    return points


class PointcloudScale(object):

  def __init__(self, lo=0.8, hi=1.25):
    self.lo, self.hi = lo, hi

  def __call__(self, points):
    scaler = np.random.uniform(self.lo, self.hi)
    points[:, 0:3] *= scaler
    return points


class PointcloudTranslate(object):

  def __init__(self, translate_range=0.1):
    self.translate_range = translate_range

  def __call__(self, points):
    translation = np.random.uniform(-self.translate_range, self.translate_range)
    points[:, 0:3] += translation
    return points


class PointcloudRotatePerturbation(object):
  """Applies a random rotation to a point cloud."""

  def __init__(self, angle_sigma=0.06, angle_clip=0.18):
    self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

  def _get_angles(self):
    angles = np.clip(
        self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
    )

    return angles

  def __call__(self, points):
    angles = self._get_angles()
    rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
    ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
    rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

    rotation_matrix = torch.matmul(torch.matmul(rz, ry), rx)

    normals = points.size(1) > 3
    if not normals:
      return torch.matmul(points, rotation_matrix.t())
    else:
      pc_xyz = points[:, 0:3]
      pc_normals = points[:, 3:]
      points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
      points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

      return points
