# GeoMatch: Geometry Matching for Multi-Embodiment Grasping

While significant progress has been made on the problem of generating grasps, many existing learning-based approaches still concentrate on a single embodiment, provide limited generalization to higher DoF end-effectors and cannot capture a diverse set of grasp modes. In this paper, we tackle the problem of grasping multi-embodiments through the viewpoint of learning rich geometric representations for both objects and end-effectors using Graph Neural Networks (GNN). Our novel method - GeoMatch - applies supervised learning on grasping data from multiple embodiments, learning end-to-end contact point likelihood maps as well as conditional autoregressive prediction of grasps keypoint-by-keypoint. We compare our method against 3 baselines that provide multi-embodiment support. Our approach performs better across 3 end-effectors, while also providing competitive diversity of grasps. Examples can be found at geomatch.github.io.

This is source code for the paper: [Geometry Matching for Multi-Embodiment Grasping](https://arxiv.org/abs/2312.03864).

## Installation

To get started, creating an Anaconda or virtual environment is recommended.

This repository was developed on pytorch 1.13 among other dependencies:

```pip install torch==1.13.1 pytorch-kinematics matplotlib transforms3d numpy scipy plotly trimesh urdf_parser_py tqdm argparse```

For this work, we used the data from [GenDexGrasp: Generalizable Dexterous Grasping](https://github.com/tengyu-liu/GenDexGrasp/tree/main). Please follow instructions on this link to download.

## Usage

To train our model, run:

```python3 train.py --epochs=XXX --batch_size=YYY```

To generate grasps for a given object and all end-effectors, run:

```python3 generate_grasps_for_obj.py --saved_model_dir=<your_trained_model> --object_name=<selected_object>```

Optionally, you can plot grasps as they're generated by passing in `--plot_grasps` to the command above.

To generate grasps for all objects of the eval set and all end-effectors, run:

```python3 generate_grasps_for_all_objs.py --saved_model_dir=<your_trained_model>```


## Citing this work

If you liked and used our repository, please cite us:

```
@inproceedings{attarian2023geometry,
  title={Geometry Matching for Multi-Embodiment Grasping},
  author={Attarian, Maria and Asif, Muhammad Adil and Liu, Jingzhou and Hari, Ruthrash and Garg, Animesh and Gilitschenski, Igor and Tompson, Jonathan},
  booktitle={Proceedings of the 7th Conference on Robot Learning (CoRL)},
  year={2023}
}
```

## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
