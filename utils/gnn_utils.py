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

"""GNN utilities."""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy.sparse as sp
from scipy.spatial import distance
import torch
from torch import nn
import trimesh as tm

colors = ['blue', 'red', 'yellow', 'pink', 'gray', 'orange']


def plot_mesh(mesh, color='lightblue', opacity=1.0):
  return go.Mesh3d(
      x=mesh.vertices[:, 0],
      y=mesh.vertices[:, 1],
      z=mesh.vertices[:, 2],
      i=mesh.faces[:, 0],
      j=mesh.faces[:, 1],
      k=mesh.faces[:, 2],
      color=color,
      opacity=opacity,
  )


def plot_point_cloud(
    pts, color='lightblue', mode='markers', colorscale='Viridis', size=3
):
  return go.Scatter3d(
      x=pts[:, 0],
      y=pts[:, 1],
      z=pts[:, 2],
      mode=mode,
      marker=dict(color=color, colorscale=colorscale, size=size),
  )


def plot_grasp(
    hand_model,
    key_points_idx_dict,
    q,
    object_name,
    obj_pc,
    obj_contact_map,
    selected_kp_idx,
    selected_vert,
    object_mesh_basedir='/data/grasp_gnn/object',
):
  """Plots a given grasp."""

  robot_keypoints_trans = hand_model.get_key_points_from_indices(
      key_points_idx_dict, q=q
  )
  vis_data = hand_model.get_plotly_data(q=q, opacity=0.5)
  object_mesh_path = os.path.join(
      object_mesh_basedir,
      f'{object_name.split("+")[0]}',
      f'{object_name.split("+")[1]}',
      f'{object_name.split("+")[1]}.stl',
  )
  vis_data += [plot_mesh(mesh=tm.load(object_mesh_path))]
  vis_data += [plot_point_cloud(obj_pc, obj_contact_map.squeeze())]
  vis_data += [plot_point_cloud(selected_vert, color='red')]
  vis_data += [plot_point_cloud(robot_keypoints_trans, color='black')]
  vis_data += [
      plot_point_cloud(
          robot_keypoints_trans[selected_kp_idx, :][None], color='red'
      )
  ]
  fig = go.Figure(data=vis_data)
  fig.show()


def plot_hand_only(hand_model, q, key_points, selected_kp):
  vis_data = hand_model.get_plotly_data(q=q, opacity=0.5)
  vis_data += [plot_point_cloud(key_points, color='black')]
  vis_data += [plot_point_cloud(selected_kp, color='red')]
  fig = go.Figure(data=vis_data)
  fig.show()


def plot_obj_only(obj_pc, obj_contact_map, selected_vert):
  vis_data = [plot_point_cloud(obj_pc, obj_contact_map.squeeze())]
  vis_data += [plot_point_cloud(selected_vert, color='red')]
  fig = go.Figure(data=vis_data)
  fig.show()


def encode_onehot(labels, threshold=0.5):
  if isinstance(labels, np.ndarray):
    labels = torch.Tensor(labels)
  labels_onehot = torch.where(labels > threshold, 1.0, 0.0)
  return labels_onehot


def euclidean_min_dist(point, point_cloud):
  dist_array = np.linalg.norm(
      np.array(point_cloud) - np.array(point).reshape((1, 3)), axis=-1
  )
  return np.min(dist_array), np.argsort(dist_array)


def generate_contact_maps(contact_map):
  labels = contact_map
  labels = torch.FloatTensor(encode_onehot(labels))
  return labels


def normalize_pc(points, scale_fact):
  centroid = torch.mean(points, dim=0)
  points -= centroid
  points /= scale_fact

  return points


def generate_adj_mat_feats(point_cloud, knn=8):
  """Generates a graph from a given point cloud based on k-NN."""

  features = sp.csr_matrix(point_cloud, dtype=np.float32)

  # build graph
  dist = distance.squareform(distance.pdist(np.asarray(point_cloud)))
  closest = np.argsort(dist, axis=1)
  adj = np.zeros(closest.shape)

  for i in range(adj.shape[0]):
    adj[i, closest[i, 0 : knn + 1]] = 1

  adj = sp.coo_matrix(adj)

  # build symmetric adjacency matrix
  adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

  # features = normalize(features)
  adj = normalize(adj + sp.eye(adj.shape[0]))

  features = torch.FloatTensor(np.array(features.todense()))
  adj = sparse_mx_to_torch_sparse_tensor(adj)

  return adj, features


def normalize(mx):
  """Row-normalize sparse matrix."""
  rowsum = np.array(mx.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.0
  r_mat_inv = sp.diags(r_inv)
  mx = r_mat_inv.dot(mx)
  return mx


def train_metrics(output, labels, threshold=0.5):
  """Generates all training metrics."""

  batch_size = labels.shape[0]
  size = labels.shape[1] * labels.shape[2]
  total_num = batch_size * size
  preds = nn.Sigmoid()(output)
  preds = encode_onehot(preds, threshold=threshold)

  true_positives = (preds * labels).sum()
  false_positives = (preds * (1 - labels)).sum()
  false_negatives = ((1 - preds) * labels).sum()
  true_negatives = ((1 - preds) * (1 - labels)).sum()

  precision = true_positives / ((true_positives + false_positives) + 1e-6)
  recall = true_positives / (true_positives + false_negatives)
  acc = (true_negatives + true_positives) / total_num
  return (
      acc,
      precision,
      recall,
      true_positives,
      true_negatives,
      false_positives,
      false_negatives,
  )


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
  """Convert a scipy sparse matrix to a torch sparse tensor."""
  sparse_mx = sparse_mx.tocoo().astype(np.float32)
  indices = torch.from_numpy(
      np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
  )
  values = torch.from_numpy(sparse_mx.data)
  shape = torch.Size(sparse_mx.shape)
  return torch.sparse.FloatTensor(indices, values, shape)


def graph_cross_convolution(inp, kernel, inp_adj, krn_adj):
  """Experimental: Graph Cross Convolution."""
  kernel = torch.matmul(krn_adj.to_dense(), kernel)
  support = torch.matmul(inp, kernel.transpose(-2, -1))
  output = torch.matmul(inp_adj.to_dense(), support)
  return output


def matplotlib_imshow(img, one_channel=False):
  if one_channel:
    img = img.mean(dim=0)
  img = img / 2 + 0.5  # unnormalize
  npimg = img.numpy()
  if one_channel:
    plt.imshow(npimg, cmap='Greys')
  else:
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def square(true_positives, false_positives, true_negatives, false_negatives):
  """Defines a precision/recall square for Tensorboard plotting."""

  fig = plt.figure(figsize=(9, 9))
  ax = fig.add_subplot(111)
  rect1 = matplotlib.patches.Rectangle((0, 2), 2, 2, color='green')
  rect2 = matplotlib.patches.Rectangle((2, 2), 2, 2, color='red')
  rect3 = matplotlib.patches.Rectangle((0, 0), 2, 2, color='red')
  rect4 = matplotlib.patches.Rectangle((2, 0), 2, 2, color='green')

  ax.add_patch(rect1)
  ax.add_patch(rect2)
  ax.add_patch(rect3)
  ax.add_patch(rect4)
  rectangles = [rect1, rect2, rect3, rect4]
  tags = [
      'True Positives=' + str(true_positives.item()),
      'False Positives=' + str(false_positives.item()),
      'True Negatives=' + str(true_negatives.item()),
      'False Negatives=' + str(false_negatives.item()),
  ]

  for r in range(4):
    ax.add_artist(rectangles[r])
    rx, ry = rectangles[r].get_xy()
    cx = rx + rectangles[r].get_width() / 2.0
    cy = ry + rectangles[r].get_height() / 2.0

    ax.annotate(
        tags[r],
        (cx, cy),
        color='w',
        weight='bold',
        fontsize=12,
        ha='center',
        va='center',
    )

  plt.xlim([0, 4])
  plt.ylim([0, 4])
  plt.savefig('square.jpg')
