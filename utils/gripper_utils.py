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

"""Utilities to represent an end-effector."""

import json
import os
import numpy as np
from plotly import graph_objects as go
import pytorch_kinematics as pk
from pytorch_kinematics.urdf_parser_py.urdf import Box
from pytorch_kinematics.urdf_parser_py.urdf import Cylinder
from pytorch_kinematics.urdf_parser_py.urdf import Mesh
from pytorch_kinematics.urdf_parser_py.urdf import Sphere
from pytorch_kinematics.urdf_parser_py.urdf import URDF
import torch
import torch.nn
import transforms3d
import trimesh as tm
import trimesh.sample
import urdf_parser_py.urdf as URDF_PARSER
from utils import math_utils


class HandModel:
  """Hand model class based on: https://github.com/tengyu-liu/GenDexGrasp/blob/main/utils_model/HandModel.py."""

  def __init__(
      self,
      robot_name,
      urdf_filename,
      mesh_path,
      batch_size=1,
      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
      hand_scale=2.0,
      data_dir='data',
  ):
    self.device = device
    self.batch_size = batch_size
    self.data_dir = data_dir

    self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(
        dtype=torch.float, device=self.device
    )
    self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)

    if robot_name == 'allegro_right':
      self.robot_name = 'allegro_right'
      robot_name = 'allegro'
    else:
      self.robot_name = robot_name

    self.global_translation = None
    self.global_rotation = None
    self.softmax = torch.nn.Softmax(dim=-1)

    self.contact_point_dict = json.load(
        open(os.path.join(self.data_dir, 'urdf/contact_%s.json' % robot_name))
    )
    self.contact_point_basis = {}
    self.contact_normals = {}
    self.surface_points = {}
    self.surface_points_normal = {}
    visual = URDF.from_xml_string(open(urdf_filename).read())
    self.centroids = json.load(
        open(os.path.join(self.data_dir, 'robot_centroids.json'))
    )[robot_name]
    self.keypoints = json.load(
        open(os.path.join(self.data_dir, 'robot_keypoints.json'))
    )[self.robot_name]
    self.key_point_idx_dict = {}
    self.mesh_verts = {}
    self.mesh_faces = {}

    self.canon_verts = []
    self.canon_faces = []
    self.idx_vert_faces = []
    self.face_normals = []
    self.links = [link.name for link in visual.links]

    for i_link, link in enumerate(visual.links):
      print(f'Processing link #{i_link}: {link.name}')

      if not link.visuals:
        continue
      if isinstance(link.visuals[0].geometry, Mesh):
        if (
            robot_name == 'shadowhand'
            or robot_name == 'allegro'
            or robot_name == 'barrett'
        ):
          filename = link.visuals[0].geometry.filename.split('/')[-1]
        elif robot_name == 'allegro':
          filename = f"{link.visuals[0].geometry.filename.split('/')[-2]}/{link.visuals[0].geometry.filename.split('/')[-1]}"
        else:
          filename = link.visuals[0].geometry.filename
        mesh = tm.load(
            os.path.join(mesh_path, filename), force='mesh', process=False
        )
      elif isinstance(link.visuals[0].geometry, Cylinder):
        mesh = tm.primitives.Cylinder(
            radius=link.visuals[0].geometry.radius,
            height=link.visuals[0].geometry.length,
        )
      elif isinstance(link.visuals[0].geometry, Box):
        mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
      elif isinstance(link.visuals[0].geometry, Sphere):
        mesh = tm.primitives.Sphere(radius=link.visuals[0].geometry.radius)
      else:
        print(type(link.visuals[0].geometry))
        raise NotImplementedError
      try:
        scale = np.array(link.visuals[0].geometry.scale).reshape([1, 3])
      except Exception:  # pylint: disable=broad-exception-caught
        scale = np.array([[1, 1, 1]])
      try:
        rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
        translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])

      except Exception:  # pylint: disable=broad-exception-caught
        rotation = transforms3d.euler.euler2mat(0, 0, 0)
        translation = np.array([[0, 0, 0]])

      if self.robot_name == 'shadowhand':
        pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=64)
        pts_normal = np.array(
            [mesh.face_normals[x] for x in pts_face_index], dtype=float
        )
      else:
        pts, pts_face_index = trimesh.sample.sample_surface(
            mesh=mesh, count=128
        )
        pts_normal = np.array(
            [mesh.face_normals[x] for x in pts_face_index], dtype=float
        )

      pts *= scale
      if robot_name == 'shadowhand':
        pts = pts[:, [0, 2, 1]]
        pts_normal = pts_normal[:, [0, 2, 1]]
        pts[:, 1] *= -1
        pts_normal[:, 1] *= -1

      pts = np.matmul(rotation, pts.T).T + translation
      pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
      pts_normal = np.concatenate(
          [pts_normal, np.ones([len(pts_normal), 1])], axis=-1
      )
      self.surface_points[link.name] = (
          torch.from_numpy(pts)
          .to(device)
          .float()
          .unsqueeze(0)
          .repeat(batch_size, 1, 1)
      )
      self.surface_points_normal[link.name] = (
          torch.from_numpy(pts_normal)
          .to(device)
          .float()
          .unsqueeze(0)
          .repeat(batch_size, 1, 1)
      )

      # visualization mesh
      self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
      if robot_name == 'shadowhand':
        self.mesh_verts[link.name] = self.mesh_verts[link.name][:, [0, 2, 1]]
        self.mesh_verts[link.name][:, 1] *= -1
      self.mesh_verts[link.name] = (
          np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
      )
      self.mesh_faces[link.name] = np.array(mesh.faces)

      # contact point
      if link.name in self.contact_point_dict:
        cpb = np.array(self.contact_point_dict[link.name])
        if len(cpb.shape) > 1:
          cpb = cpb[np.random.randint(cpb.shape[0], size=1)][0]

        cp_basis = mesh.vertices[cpb] * scale
        if robot_name == 'shadowhand':
          cp_basis = cp_basis[:, [0, 2, 1]]
          cp_basis[:, 1] *= -1
        cp_basis = np.matmul(rotation, cp_basis.T).T + translation
        cp_basis = torch.cat(
            [
                torch.from_numpy(cp_basis).to(device).float(),
                torch.ones([4, 1]).to(device).float(),
            ],
            dim=-1,
        )
        self.contact_point_basis[link.name] = cp_basis.unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        v1 = cp_basis[1, :3] - cp_basis[0, :3]
        v2 = cp_basis[2, :3] - cp_basis[0, :3]
        v1 = v1 / torch.norm(v1)
        v2 = v2 / torch.norm(v2)
        self.contact_normals[link.name] = torch.cross(v1, v2).view([1, 3])
        self.contact_normals[link.name] = (
            self.contact_normals[link.name]
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

    self.scale = hand_scale

    # new 2.1
    self.revolute_joints = []
    for i, _ in enumerate(self.robot_full.joints):
      if self.robot_full.joints[i].joint_type == 'revolute':
        self.revolute_joints.append(self.robot_full.joints[i])
    self.revolute_joints_q_mid = []
    self.revolute_joints_q_var = []
    self.revolute_joints_q_upper = []
    self.revolute_joints_q_lower = []
    for i, _ in enumerate(self.robot.get_joint_parameter_names()):
      for j, _ in enumerate(self.revolute_joints):
        if (
            self.revolute_joints[j].name
            == self.robot.get_joint_parameter_names()[i]
        ):
          joint = self.revolute_joints[j]
          assert joint.name == self.robot.get_joint_parameter_names()[i]
          self.revolute_joints_q_mid.append(
              (joint.limit.lower + joint.limit.upper) / 2
          )
          self.revolute_joints_q_var.append(
              ((joint.limit.upper - joint.limit.lower) / 2) ** 2
          )
          self.revolute_joints_q_lower.append(joint.limit.lower)
          self.revolute_joints_q_upper.append(joint.limit.upper)

    joint_lower = np.array(self.revolute_joints_q_lower)
    joint_upper = np.array(self.revolute_joints_q_upper)
    joint_mid = (joint_lower + joint_upper) / 2
    joints_q = (joint_mid + joint_lower) / 2
    self.rest_pose = (
        torch.from_numpy(
            np.concatenate([np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]), joints_q])
        )
        .unsqueeze(0)
        .to(device)
        .float()
    )

    self.revolute_joints_q_lower = (
        torch.Tensor(self.revolute_joints_q_lower)
        .repeat([self.batch_size, 1])
        .to(device)
    )
    self.revolute_joints_q_upper = (
        torch.Tensor(self.revolute_joints_q_upper)
        .repeat([self.batch_size, 1])
        .to(device)
    )

    self.rest_pose = self.rest_pose.repeat([self.batch_size, 1])

    self.current_status = None
    self.canonical_keypoints = self.get_canonical_keypoints().to(device)

  def update_kinematics(self, q):
    self.global_translation = q[:, :3]

    self.global_rotation = (
        math_utils.robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
    )
    self.current_status = self.robot.forward_kinematics(q[:, 9:])

  def get_surface_points(self, q=None, downsample=False):
    """Returns surface points on the end-effector on a given pose."""

    if q is not None:
      self.update_kinematics(q)
    surface_points = []
    for link_name in self.surface_points:
      if self.robot_name == 'robotiq_3finger' and link_name == 'gripper_palm':
        continue
      if (
          self.robot_name == 'robotiq_3finger_real_robot'
          and link_name == 'palm'
      ):
        continue
      trans_matrix = self.current_status[link_name].get_matrix()
      surface_points.append(
          torch.matmul(
              trans_matrix, self.surface_points[link_name].transpose(1, 2)
          ).transpose(1, 2)[..., :3]
      )
    surface_points = torch.cat(surface_points, 1)
    surface_points = torch.matmul(
        self.global_rotation, surface_points.transpose(1, 2)
    ).transpose(1, 2) + self.global_translation.unsqueeze(1)
    if downsample:
      surface_points = surface_points[
          :, torch.randperm(surface_points.shape[1])
      ][:, :1000]
    return surface_points * self.scale

  def get_canonical_keypoints(self):
    """Returns canonical keypoints aka the N user-selected keypoints."""

    self.update_kinematics(self.rest_pose)
    key_points = np.array([
        np.array(keypoint[str(i)][0])
        for i, keypoint in enumerate(self.keypoints)
    ])
    key_points = torch.tensor(key_points).unsqueeze(0).float().to(self.device)
    key_points = key_points.repeat(self.batch_size, 1, 1)
    key_points -= self.global_translation.unsqueeze(1)
    key_points = torch.matmul(
        torch.inverse(self.global_rotation), key_points.transpose(1, 2)
    ).transpose(1, 2)
    new_key_points = []

    for i, keypoint in enumerate(self.keypoints):
      curr_keypoint = key_points[0, i, :]
      curr_keypoint_link_name = keypoint[str(i)][1]
      curr_keypoint = torch.cat(
          (curr_keypoint.clone().detach(), torch.tensor([1.0]).to(self.device))
      ).float()

      trans_matrix = self.current_status[curr_keypoint_link_name].get_matrix()
      # Address batch size if present.
      if trans_matrix.shape[0] != self.batch_size:
        trans_matrix = trans_matrix.repeat(self.batch_size, 1, 1)

      self.key_point_idx_dict[curr_keypoint_link_name] = []
      new_key_points.append(
          torch.matmul(
              torch.inverse(trans_matrix),
              curr_keypoint[None, None].transpose(1, 2),
          ).transpose(1, 2)[..., :3]
      )
    new_key_points = torch.cat(new_key_points, 1)

    return new_key_points

  def get_static_key_points(self, q, surface_pt_sample=None):
    """Returns the canonical keypoints when in a given end-effector pose."""

    final_key_points = []
    final_key_points_idx = []

    self.update_kinematics(q)

    for i in range(self.canonical_keypoints.shape[1]):
      curr_keypoint = self.canonical_keypoints[0, i, :]
      curr_keypoint_link_name = self.keypoints[i][str(i)][1]
      curr_keypoint = torch.cat(
          (curr_keypoint.clone().detach(), torch.tensor([1.0]).to(self.device))
      ).float()

      if surface_pt_sample is not None:
        self.key_point_idx_dict[curr_keypoint_link_name] = []

      trans_matrix = self.current_status[curr_keypoint_link_name].get_matrix()
      final_key_points.append(
          torch.matmul(
              trans_matrix, curr_keypoint[None, None].transpose(1, 2)
          ).transpose(1, 2)[..., :3]
      )

    final_key_points = torch.cat(final_key_points, 1)
    final_key_points = torch.matmul(
        self.global_rotation, final_key_points.transpose(1, 2)
    ).transpose(1, 2) + self.global_translation.unsqueeze(1)
    final_key_points = (final_key_points * self.scale).squeeze(0)

    if surface_pt_sample is not None:
      for i, final_kp in enumerate(final_key_points):
        curr_keypoint_link_name = self.keypoints[i][str(i)][1]
        closest_vert_idx = np.argsort(
            np.linalg.norm(
                np.array(self.mesh_verts[curr_keypoint_link_name])
                - np.array(final_kp).reshape((1, 3)),
                axis=-1,
            )
        )[0]
        self.key_point_idx_dict[curr_keypoint_link_name].append(
            torch.tensor(closest_vert_idx)
        )
        closest_surface_sample_idx = np.argsort(
            np.linalg.norm(
                np.array(surface_pt_sample)
                - np.array(final_kp).reshape((1, 3)),
                axis=-1,
            )
        )[0]
        final_key_points_idx.append(closest_surface_sample_idx)

      for k in self.key_point_idx_dict:
        self.key_point_idx_dict[k] = torch.tensor(self.key_point_idx_dict[k])

    return (
        final_key_points,
        self.key_point_idx_dict,
        torch.Tensor(final_key_points_idx),
    )

  def get_key_points_from_indices(self, key_point_idx_dict, q=None):
    """Returns keypoints from a set of indices when in a given pose."""

    if q is not None:
      self.update_kinematics(q)

    key_points = []
    for link_name in key_point_idx_dict:
      trans_matrix = self.current_status[link_name].get_matrix()
      pts = np.concatenate(
          [
              self.mesh_verts[link_name],
              np.ones([len(self.mesh_verts[link_name]), 1]),
          ],
          axis=-1,
      )
      surface_points = (
          torch.from_numpy(pts).float().unsqueeze(0).repeat(1, 1, 1)
      )

      key_point_idx = key_point_idx_dict[link_name]
      key_points.append(
          torch.matmul(
              trans_matrix,
              surface_points[:, key_point_idx.long(), :].transpose(1, 2),
          ).transpose(1, 2)
      )

    key_points = torch.cat(key_points, dim=1)[..., :3]
    key_points = torch.matmul(
        self.global_rotation, key_points.transpose(1, 2)
    ).transpose(1, 2) + self.global_translation.unsqueeze(1)
    return (key_points * self.scale).squeeze(0)

  def get_meshes_from_q(self, q=None, i=0):
    """Returns gripper meshes in a given pose."""

    data = []
    if q is not None:
      self.update_kinematics(q)
    for _, link_name in enumerate(self.mesh_verts):
      trans_matrix = self.current_status[link_name].get_matrix()
      trans_matrix = (
          trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
      )
      v = self.mesh_verts[link_name]
      transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
      transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
      transformed_v = np.matmul(
          self.global_rotation[i].detach().cpu().numpy(), transformed_v.T
      ).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
      transformed_v = transformed_v * self.scale
      f = self.mesh_faces[link_name]
      data.append(tm.Trimesh(vertices=transformed_v, faces=f))
    return data

  def get_plotly_data(self, q=None, i=0, color='lightblue', opacity=1.0):
    """Returns plot data for the gripper in a given pose."""

    data = []
    if q is not None:
      self.update_kinematics(q)
    for _, link_name in enumerate(self.mesh_verts):
      trans_matrix = self.current_status[link_name].get_matrix()
      trans_matrix = (
          trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
      )
      v = self.mesh_verts[link_name]
      transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
      transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
      transformed_v = np.matmul(
          self.global_rotation[i].detach().cpu().numpy(), transformed_v.T
      ).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
      transformed_v = transformed_v * self.scale
      f = self.mesh_faces[link_name]
      data.append(
          go.Mesh3d(
              x=transformed_v[:, 0],
              y=transformed_v[:, 1],
              z=transformed_v[:, 2],
              i=f[:, 0],
              j=f[:, 1],
              k=f[:, 2],
              color=color,
              opacity=opacity,
          )
      )
    return data
