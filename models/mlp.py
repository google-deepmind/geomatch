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

"""Implementation of a Multi-Layer Perceptron."""

import copy
from torch import nn
import torch.nn.functional as F


def clones(module, n):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MLP(nn.Module):
  """MLP class."""

  def __init__(self, in_features, out_features, num_hidden, hidden_dim) -> None:
    super().__init__()

    self.layer0 = nn.Linear(in_features, hidden_dim)
    self.layers = clones(nn.Linear(hidden_dim, hidden_dim), num_hidden)
    self.out = nn.Linear(hidden_dim, out_features)

  def forward(self, x):
    x = F.relu(self.layer0(x))

    for l in self.layers:
      x = F.relu(l(x))

    return self.out(x)
