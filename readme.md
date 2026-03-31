# HOPE + Dual-Model Parking Framework

![pipeline](assets/algo_struct.png)

本仓库基于原始论文 **HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios**，在其上扩展了一套用于狭窄车位泊车的双模型框架，并补齐了训练、评测、分析和论文出图脚本。

这份 README 的目标不是介绍基础强化学习，而是让下一位开发者、合作者或 AI 能在最短时间内搞清楚：

- 项目做了什么
- 关键代码在哪里
- 双模型是怎么工作的
- 如何训练、评测、出图
- 当前论文里使用的结果口径是什么

## 1. 项目概览

当前仓库包含两条主线：

- 单模型 HOPE：原始前向泊入策略
- 双模型扩展：前向 HOPE + 反向泊出模型 + Stall-aware RS 连接

双模型的核心想法是：

1. 正常情况下，仍然优先使用前向 HOPE 直接泊入。
2. 如果车辆在狭窄车位附近明显卡顿，再激活双模型连接。
3. 从目标车位出发，用泊出模型在克隆环境中反向 rollout，生成一串锚点。
4. 当前向策略卡住时，用 Reeds-Shepp 路径把当前状态连接到某个反向锚点。
5. 连接成功后，执行 `forward policy -> RS connector -> reverse replay -> park`。

这套机制主要是为 `Complex` 和 `Extrem` 难度下的窄位场景设计的。

## 2. 当前论文口径

当前论文中使用的“成功率”分两类：

- 单模型 HOPE：`final parking success rate`
- 双模型：`path planning success rate`

这里的双模型 `path planning success rate` 定义为：

- 最终真正停进车位，记成功
- 或虽然最终没有停进去，但已经成功连上并生成可执行连接路径，也记成功

这是当前论文图表里统一采用的口径。它和脚本中原生的 `final_parking_success_rate` 不是同一个量。

当前主结果总结如下：

| Scene | HOPE | Dual |
|---|---:|---:|
| Normal | 99.6% | 99.8% |
| Complex | 97.1% | 98.8% |
| Extrem | 93.8% | 98.4% |
| dlp | 94.0% | 95.35% |

按车位类型拆分后的当前可用结果：

| Slot Type | HOPE | Dual |
|---|---:|---:|
| Normal Bay | 100.0% | 100.0% |
| Normal Parallel | 100.0% | 100.0% |
| Complex Bay | 98.0% | 100.0% |
| Complex Parallel | 95.0% | 98.0% |
| Extrem Parallel | 93.8% | 98.4% |
| dlp (Bay-like) | 94.0% | 95.35% |

注意：

- `Extrem Bay` 在当前 benchmark 中没有正式定义。
- `dlp` 是固定数据集，不是随机 `Bay/Parallel` 生成场景；当前统计里它全部是 `Bay-like`。

## 3. 环境准备

建议使用已有的 `conda` 环境 `pytorch`。

```bash
cd /path/to/HOPE
conda run -n pytorch python -V
conda run -n pytorch pip install -r requirements.txt
```

如果是无界面服务器，建议统一使用 headless 环境变量：

```bash
export QT_QPA_PLATFORM=offscreen
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
```

## 4. 仓库结构

最重要的目录如下：

- `src/model/ckpt/`
  单模型预训练权重
- `src/model/agent/`
  策略代理、RS 执行器、双模型连接逻辑
- `src/env/`
  环境、地图、动作掩码、泊出环境
- `src/train/`
  训练脚本
- `src/evaluation/`
  评测脚本
- `src/analysis/`
  论文图和案例导出脚本
- `src/log/exp/`
  训练输出
- `src/log/eval/`
  评测输出
- `src/log/analysis/`
  案例图、轨迹图、热力图
- `src/log/paper_support/`
  论文结果图包和汇总 JSON

## 5. 关键代码入口

如果你只看几个文件，优先看这些：

- `src/model/agent/bidirectional_parking_agent.py`
  双模型主逻辑。包含：
  - 反向锚点生成
  - stall 检测
  - 连接门控
  - RS 连接
  - reverse replay
- `src/model/agent/parking_agent.py`
  前向策略与 RS 执行器的融合逻辑
- `src/model/action_mask.py`
  动作掩码。基于 LiDAR 和车辆几何过滤危险离散动作
- `src/env/car_parking_out_base.py`
  泊出环境。成功条件为车辆与原车位 IoU 小于阈值，并带分段 IoU 奖励
- `src/env/env_wrapper.py`
  observation、reward shaping、action rescale
- `src/train/train_HOPE_unpark_ppo.py`
  泊出模型训练入口
- `src/evaluation/eval_extrem_bidirectional.py`
  双模型评测入口

## 6. 输入、输出与动作掩码

当前策略输入包含：

- `LiDAR`
- `target pose`
- `bird's-eye image`
- `action mask`

当前环境配置中：

- `USE_LIDAR = True`
- `USE_IMG = True`
- `USE_ACTION_MASK = True`

动作掩码模块会：

- 预计算离散动作下的车辆占据盒
- 基于当前 LiDAR 观测估算每个动作的安全步长
- 抑制会导致碰撞或危险穿模的动作

对于论文总框架图，动作掩码值得单独画成一个小模块，因为它是 HOPE 与纯连续控制器的一个非常重要区别点。

## 7. 现有权重

单模型权重：

- `src/model/ckpt/HOPE_SAC0.pt`
- `src/model/ckpt/HOPE_SAC1.pt`
- `src/model/ckpt/HOPE_PPO.pt`

双模型中的泊出模型常见路径：

- `src/log/exp/unpark_ppo_<timestamp>/PPO_unpark_best.pt`

当前论文相关结果主要使用：

- 前向泊入：`HOPE_SAC0.pt`
- 反向泊出：`PPO_unpark_best.pt`

## 8. 单模型评测

官方混合场景评测：

```bash
cd /path/to/HOPE
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy MPLBACKEND=Agg \
conda run -n pytorch python src/evaluation/eval_mix_scene.py \
  src/model/ckpt/HOPE_SAC0.pt \
  --eval_episode 2000 \
  --visualize False
```

说明：

- `--eval_episode 2000` 表示每个场景 2000 次，不是总共 2000 次
- 输出写到 `src/log/eval/`

## 9. 泊出模型训练

泊出模型训练入口：

- `src/train/train_HOPE_unpark_ppo.py`

当前默认配置：

- 场景：`Complex,Extrem`
- 支持多环境并行：`--num_envs`
- 当前经验上推荐 `16` 环境

训练：

```bash
cd /path/to/HOPE
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy MPLBACKEND=Agg \
conda run -n pytorch python src/train/train_HOPE_unpark_ppo.py \
  --train_episode 50000 \
  --eval_episode 200 \
  --levels Complex,Extrem \
  --num_envs 16 \
  --visualize False \
  --verbose True
```

继续训练：

```bash
cd /path/to/HOPE
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy MPLBACKEND=Agg \
conda run -n pytorch python src/train/train_HOPE_unpark_ppo.py \
  --agent_ckpt /path/to/PPO_unpark_best.pt \
  --train_episode 50000 \
  --eval_episode 200 \
  --levels Complex,Extrem \
  --num_envs 16 \
  --visualize False \
  --verbose True
```

输出目录示例：

- `src/log/exp/unpark_ppo_<timestamp>/`

关键文件：

- `PPO_unpark_best.pt`
- `best.txt`
- `reward.png`
- `events.out.tfevents.*`

## 10. 双模型评测

当前正式入口：

- `src/evaluation/eval_extrem_bidirectional.py`

示例：

```bash
cd /path/to/HOPE
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy MPLBACKEND=Agg \
conda run -n pytorch python src/evaluation/eval_extrem_bidirectional.py \
  src/model/ckpt/HOPE_SAC0.pt \
  /path/to/PPO_unpark_best.pt \
  --eval_episode 2000 \
  --visualize False \
  --verbose False
```

输出目录示例：

- `src/log/eval/bidirectional_extrem_<timestamp>/`

结果文件：

- `summary.json`
- `episode_details.json`

常见字段说明：

- `planning_success_rate`
  脚本原生定义的规划相关成功率
- `final_parking_success_rate`
  最终真正停车成功率
- `connection_rate`
  触发连接的比例
- `avg_step_num`
- `avg_total_path_length`
- `avg_total_gear_shifts`

## 11. 当前双模型连接机制

当前实现已经不是“只要能连就立刻接管”的旧版本，而是：

- 前向优先
- 当前向明显卡顿时才尝试连接
- 简单场景尽量不接管
- 若 `connection_used = false`，双模型前向部分应尽量与单模型保持一致

连接链路是：

1. 在克隆环境中用泊出模型从目标车位反向 rollout，生成 `reverse_states`
2. 当前向策略卡顿时，搜索可达反向锚点
3. 用 RS 路径连接到锚点
4. 执行 reverse replay 回到车位

## 12. 出图脚本总览

### 12.1 固定场景对比图

脚本：

- `src/analysis/export_vector_case_plots.py`

用途：

- 导出单模型 / 双模型对比轨迹
- 支持 `line` 风格和 `footprints` 风格
- 支持固定 `scene:seed`

示例：

```bash
cd /path/to/HOPE
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy MPLBACKEND=Agg \
conda run -n pytorch python src/analysis/export_vector_case_plots.py \
  --forward_ckpt src/model/ckpt/HOPE_SAC0.pt \
  --unpark_ckpt /path/to/PPO_unpark_best.pt \
  --scene Normal:11850 \
  --scene Extrem:23 \
  --png_dpi 600 \
  --verbose True
```

### 12.2 低换挡案例图

脚本：

- `src/analysis/build_low_shift_case_gallery.py`

用途：

- 挑出轨迹干净、换挡少的双模型案例
- 适合论文中“qualitative results”展示

### 12.3 相对车位热力图

脚本：

- `src/analysis/build_relative_slot_heatmaps.py`

用途：

- 以目标车位为局部坐标系原点
- 统计动作主要耗费在哪些相对区域
- 用于说明单模型在复杂场景中大量动作浪费在车位外

示例：

```bash
cd /path/to/HOPE
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy MPLBACKEND=Agg \
conda run -n pytorch python src/analysis/build_relative_slot_heatmaps.py \
  --forward_ckpt src/model/ckpt/HOPE_SAC0.pt \
  --unpark_ckpt /path/to/PPO_unpark_best.pt \
  --level Complex \
  --max_seed 200 \
  --selection_mode single_hard \
  --min_single_steps 35 \
  --min_step_gain 15 \
  --png_dpi 360
```

### 12.4 论文图包

脚本：

- `src/analysis/build_dual_framework_paper_package.py`

用途：

- 成功率图
- Bay / Parallel 图
- dual breakdown 图
- rescue cases 图
- 规划结果图
- 热力图

### 12.5 更新后的成功率总图

脚本：

- `src/analysis/plot_updated_success_summary.py`

用途：

- 输出当前论文中使用的成功率总图
- 支持 `paper` 和 `ieee` 两种主题
- 当前 `Complex` 双模型成功率可通过 `--complex-dual` 手动覆盖，例如 `98.8`

示例：

```bash
cd /path/to/HOPE
conda run -n pytorch python src/analysis/plot_updated_success_summary.py \
  --complex-dual 98.8 \
  --style ieee \
  --dpi 480
```

输出目录示例：

- `src/log/paper_support/updated_success_summary_ieee_<timestamp>/`

## 13. 常用结果目录

训练结果：

- `src/log/exp/`

评测结果：

- `src/log/eval/`

案例图与分析：

- `src/log/analysis/`

论文图：

- `src/log/paper_support/`

当前 `src/log/` 内容很多，不建议纳入 Git；仓库的 `.gitignore` 已默认忽略这些实验输出。

## 14. 已知注意事项

- 当前 benchmark 没有正式定义 `Extrem Bay`
- `dlp` 不能按 `Bay / Parallel` 拆成两类
- 无图形环境下运行时，建议统一设置 headless 变量
- 论文里如果使用“路径规划成功率”字样，必须明确其定义，不要与 `final_parking_success_rate` 混淆
- 某些论文图脚本默认读取 `src/log/eval/` 中已有 JSON，如果你更换 checkpoint，需要重新评测或修改脚本头部路径

## 15. 如果你要快速接手这个项目

建议按这个顺序阅读：

1. `src/model/agent/bidirectional_parking_agent.py`
2. `src/env/car_parking_out_base.py`
3. `src/model/action_mask.py`
4. `src/evaluation/eval_extrem_bidirectional.py`
5. `src/analysis/plot_updated_success_summary.py`
6. `src/analysis/export_vector_case_plots.py`

建议先做这三件事验证环境没问题：

1. 跑一次单模型 `eval_mix_scene.py`
2. 跑一次双模型 `eval_extrem_bidirectional.py`
3. 跑一次 `plot_updated_success_summary.py`

如果这三步都通了，后面的训练、评测和论文图生成通常也不会有大问题。

## 16. Citation

如果你使用了原始 HOPE 工作，请引用：

```bibtex
@article{jiang2024hope,
  title={HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios},
  author={Jiang, Mingyang and Li, Yueyuan and Zhang, Songan and Chen, Siyuan and Wang, Chunxiang and Yang, Ming},
  journal={arXiv preprint arXiv:2405.20579},
  year={2024}
}
```
