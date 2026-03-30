# HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios
![pipeline](assets/algo_struct.png)

This repository contains code for the paper [HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios](https://arxiv.org/abs/2405.20579). This work proposes a novel solution to the path-planning task in parking scenarios. The planner integrates a reinforcement learning agent with Reeds-Shepp curves, enabling effective planning across diverse scenarios. HOPE guides the exploration of the reinforcement learning agent by applying an action mask mechanism and employs a transformer to integrate the perceived environmental information with the mask. Our approach achieved higher planning success rates compared with typical rule-based algorithms and traditional reinforcement learning methods, especially in challenging cases.

## Examples
### Simulation cases
![simulation](assets/examples.jpg)

### Realworld demo
[https://www.youtube.com/watch?v=62w9qhjIuRI](https://www.youtube.com/watch?v=62w9qhjIuRI)
![realworld](assets/realworld-cases.jpg)

## Setup
1. Install conda or miniconda

2. Clone the repo and build the environment
```Shell
git clone https://github.com/jiamiya/HOPE.git
cd HOPE
conda create -n HOPE python==3.8
conda activate HOPE
pip3 install -r requirements.txt
```
and install pytorch from [https://pytorch.org/](https://pytorch.org/).

## Usage
### Run a pre-trained agent
```Shell
cd src
python ./evaluation/eval_mix_scene.py ./model/ckpt/HOPE_SAC0.pt --eval_episode 10 --visualize True
```
You can find some other pre-trained weights in ``./src/model/ckpt``.

### Train the HOPE planner
```Shell
cd src
python ./train/train_HOPE_sac.py
```
or
```Shell
python ./train/train_HOPE_ppo.py
```

## Citation
If you find our work useful, please cite us as
```bibtex
@article{jiang2024hope,
  title={HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios},
  author={Jiang, Mingyang and Li, Yueyuan and Zhang, Songan and Chen, Siyuan and Wang, Chunxiang and Yang, Ming},
  journal={arXiv preprint arXiv:2405.20579},
  year={2024}
}
```


当前仓库里没有 best.pt，这次我按仓库默认预训练权重 HOPE_SAC0.pt 跑了官方口径的全量评估：4 个场景各 2000 episode，headless 模式，累计约 4212s，也就是约 70.2 分钟。

结果如下：

Extrem: 0.9380，1876 / 2000
dlp: 0.9400，1880 / 2000
Complex: 0.9710，1942 / 2000
Normal: 0.9960，1992 / 2000
总体平均成功率: 0.96125
结果已写到 minimal_full_sac0_20260322_183643.json。

如果你说的 best.pt 是别的 checkpoint，把准确路径给我，我可以按同样口径再跑一遍。