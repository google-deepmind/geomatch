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

"""Config file with parameters."""

obj_pc_n = 2048
robot_pc_n = 6
keypoint_n = 6

hidden_n = 256
obj_in_feats = 3
robot_in_feats = obj_in_feats
obj_out_feats = 512
robot_out_feats = obj_out_feats
robot_weighting = 500.0
matchnet_weighting = 200.0
num_hidden = 3

dist_loss_weight = 0.5
match_loss_weight = 0.5
