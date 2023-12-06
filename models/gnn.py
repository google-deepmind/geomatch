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

"""Implementation of Graph Convolutional Neural Networks."""

import copy
import math
import torch
from torch import nn
import torch.nn.functional as F


def clones(module, n):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class GraphConvolution(nn.Module):
  """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

  def __init__(self, in_features, out_features, bias=True):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
    if bias:
      self.bias = nn.Parameter(torch.FloatTensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)

  def forward(self, inp, adj):
    support = torch.matmul(inp, self.weight)
    output = torch.matmul(adj.to_dense(), support)
    if self.bias is not None:
      return output + self.bias
    else:
      return output

  def __repr__(self):
    return (
        self.__class__.__name__
        + ' ('
        + str(self.in_features)
        + ' -> '
        + str(self.out_features)
        + ')'
    )


class GCN(nn.Module):
  """Graph Convolutional Neural Network class."""

  def __init__(self, nfeat, nhid, nout, dropout, num_hidden):
    super().__init__()

    self.gc0 = GraphConvolution(nfeat, nhid)
    self.gc_layers = clones(GraphConvolution(nhid, nhid), num_hidden)
    self.out = nn.Linear(nhid, nout)
    self.dropout = dropout

  def forward(self, x, adj):
    x = F.relu(self.gc0(x, adj))

    for i, _ in enumerate(self.gc_layers):
      x = F.relu(self.gc_layers[i](x, adj))
    return self.out(x)
