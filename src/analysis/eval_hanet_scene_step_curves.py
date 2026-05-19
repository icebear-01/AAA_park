# -*- coding: utf-8 -*-
import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


CHECKPOINT_RE = re.compile(r"^PPO2_(\d+)\.pt$")


def _add_src_root(src_root: Path) -> None:
    src = str(src_root.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def _normalize_ckpt_step(raw_step: int) -> float:
    if raw_step % 10000 == 9999:
        return float(raw_step + 1)
    return float(raw_step)


def _discover_checkpoints(ckpt_dir: Path) -> list:
    rows = []
    for path in ckpt_dir.glob("PPO2_*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        raw_step = int(match.group(1))
        rows.append(
            {
                "path": str(path.resolve()),
                "ckpt_name": path.name,
                "raw_step": raw_step,
                "train_step": _normalize_ckpt_step(raw_step),
                "train_step_x1e4": _normalize_ckpt_step(raw_step) / 1e4,
            }
        )
    rows.sort(key=lambda item: item["raw_step"])
    return rows


def _build_agent_and_env(src_root: Path, ckpt_path: str):
    _add_src_root(src_root)
    os.environ.setdefault("HOPE_HEADLESS", "1")

    import torch
    from configs import ACTOR_CONFIGS, CRITIC_CONFIGS, USE_ACTION_MASK, USE_IMG, USE_LIDAR
    from env.car_parking_base import CarParking
    from env.env_wrapper import CarParkingWrapper
    from env.vehicle import VALID_SPEED
    from model.agent.parking_agent import ParkingAgent, RsPlanner
    from model.agent.ppo_agent import PPOAgent as PPO

    raw_env = CarParking(
        render_mode="rgb_array",
        fps=0,
        verbose=False,
        use_lidar_observation=USE_LIDAR,
        use_img_observation=USE_IMG,
        use_action_mask=USE_ACTION_MASK,
    )
    env = CarParkingWrapper(raw_env)
    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": ACTOR_CONFIGS,
        "critic_layers": CRITIC_CONFIGS,
    }
    rl_agent = PPO(configs)
    rl_agent.load(ckpt_path, params_only=True)
    step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
    agent = ParkingAgent(rl_agent, RsPlanner(step_ratio))
    return env, agent, torch


def _eval_scene(env, agent, torch_mod, level: str, episodes: int):
    from env.vehicle import Status

    success_steps = []
    all_steps = []
    success_count = 0
    env.set_level(level)

    for _ in range(episodes):
        obs = env.reset()
        agent.reset()
        done = False
        step_num = 0
        last_obs = obs["target"]

        with torch_mod.no_grad():
            while not done:
                step_num += 1
                action, _ = agent.choose_action(obs)
                if (last_obs == obs["target"]).all():
                    action = env.action_space.sample()
                last_obs = obs["target"]
                obs, _, done, info = env.step(action)
                if info["path_to_dest"] is not None:
                    agent.set_planner_path(info["path_to_dest"])

        all_steps.append(200 if info["status"] == Status.OUTBOUND else step_num)
        if info["status"] == Status.ARRIVED:
            success_count += 1
            success_steps.append(step_num)

    return {
        "episodes": int(episodes),
        "num_success": int(success_count),
        "success_rate": float(success_count / max(episodes, 1)),
        "avg_success_step": float(np.mean(success_steps)) if success_steps else math.nan,
        "std_success_step": float(np.std(success_steps)) if success_steps else math.nan,
        "avg_all_step": float(np.mean(all_steps)) if all_steps else math.nan,
        "std_all_step": float(np.std(all_steps)) if all_steps else math.nan,
    }


def _evaluate_checkpoint(task: dict) -> dict:
    src_root = Path(task["src_root"])
    ckpt_path = task["ckpt_path"]
    seed = int(task["seed"])
    episodes = int(task["episodes"])

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_hanet_step_eval_%s" % os.getpid())
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    np.random.seed(seed)
    random.seed(seed)

    env, agent, torch_mod = _build_agent_and_env(src_root, ckpt_path)
    try:
        torch_mod.set_num_threads(1)
        torch_mod.set_num_interop_threads(1)
    except Exception:
        pass
    torch_mod.manual_seed(seed)

    started = time.time()
    normal = _eval_scene(env, agent, torch_mod, "Normal", episodes)
    torch_mod.manual_seed(seed + 1)
    np.random.seed(seed + 1)
    random.seed(seed + 1)
    complex_result = _eval_scene(env, agent, torch_mod, "Complex", episodes)
    env.close()

    return {
        "ckpt_path": ckpt_path,
        "elapsed_sec": float(time.time() - started),
        "normal": normal,
        "complex": complex_result,
    }


def _plot_curve(df: pd.DataFrame, output_path: Path, value_columns: list, labels: list, colors: list, ylabel: str, show_legend: bool) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 22
    plt.rcParams["ytick.labelsize"] = 22
    plt.rcParams["axes.linewidth"] = 1.2

    fig, ax = plt.subplots(figsize=(9.8, 6.2), dpi=400)
    x = df["train_step_x1e4"].to_numpy()
    for column, label, color in zip(value_columns, labels, colors):
        ax.plot(x, df[column].to_numpy(), color=color, linewidth=3.0, marker="o", markersize=6.0, label=label)
    ax.set_xlabel("训练轮次（×10^4）")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=float(np.nanmin(x)), right=float(np.nanmax(x)))
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis="both", which="major", labelsize=22)
    if show_legend:
        ax.legend(frameon=False, fontsize=18, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpts = _discover_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise SystemExit("No PPO2 checkpoints found in %s" % args.ckpt_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "partial_results.jsonl"
    if partial_path.exists():
        partial_path.unlink()

    tasks = []
    for index, item in enumerate(ckpts):
        tasks.append(
            {
                "src_root": str(args.src_root),
                "ckpt_path": item["path"],
                "seed": args.seed + index * 97,
                "episodes": args.episodes,
            }
        )

    results_by_path = {}
    if args.workers <= 1:
        for task in tasks:
            result = _evaluate_checkpoint(task)
            results_by_path[result["ckpt_path"]] = result
            with partial_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print("done", Path(result["ckpt_path"]).name, "elapsed_sec=%.2f" % result["elapsed_sec"], flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(_evaluate_checkpoint, task): task["ckpt_path"] for task in tasks}
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                results_by_path[result["ckpt_path"]] = result
                with partial_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print("done", Path(result["ckpt_path"]).name, "elapsed_sec=%.2f" % result["elapsed_sec"], flush=True)

    rows = []
    for item in ckpts:
        result = results_by_path[item["path"]]
        row = {
            "ckpt_name": item["ckpt_name"],
            "ckpt_path": item["path"],
            "raw_step": item["raw_step"],
            "train_step": item["train_step"],
            "train_step_x1e4": item["train_step_x1e4"],
            "normal_success_rate": result["normal"]["success_rate"],
            "normal_avg_success_step": result["normal"]["avg_success_step"],
            "normal_std_success_step": result["normal"]["std_success_step"],
            "normal_avg_all_step": result["normal"]["avg_all_step"],
            "complex_success_rate": result["complex"]["success_rate"],
            "complex_avg_success_step": result["complex"]["avg_success_step"],
            "complex_std_success_step": result["complex"]["std_success_step"],
            "complex_avg_all_step": result["complex"]["avg_all_step"],
            "elapsed_sec": result["elapsed_sec"],
            "episodes_per_scene": args.episodes,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "scene_step_eval.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "src_root": str(args.src_root.resolve()),
        "ckpt_dir": str(args.ckpt_dir.resolve()),
        "episodes_per_scene": int(args.episodes),
        "workers": int(args.workers),
        "metric_primary": "avg_success_step",
        "normal_final_avg_success_step": float(df["normal_avg_success_step"].iloc[-1]),
        "complex_final_avg_success_step": float(df["complex_avg_success_step"].iloc[-1]),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    combined_path = args.output_dir / "scene_step_curve_normal_complex_avg_success_x1e4_dpi400.png"
    normal_path = args.output_dir / "scene_step_curve_normal_avg_success_x1e4_dpi400.png"
    complex_path = args.output_dir / "scene_step_curve_complex_avg_success_x1e4_dpi400.png"

    _plot_curve(
        df,
        combined_path,
        ["normal_avg_success_step", "complex_avg_success_step"],
        ["简单场景", "复杂场景"],
        ["#009E73", "#D55E00"],
        "平均完成步数",
        True,
    )
    _plot_curve(
        df,
        normal_path,
        ["normal_avg_success_step"],
        ["简单场景"],
        ["#009E73"],
        "平均完成步数",
        False,
    )
    _plot_curve(
        df,
        complex_path,
        ["complex_avg_success_step"],
        ["复杂场景"],
        ["#D55E00"],
        "平均完成步数",
        False,
    )

    print(csv_path)
    print(summary_path)
    print(combined_path)
    print(normal_path)
    print(complex_path)


if __name__ == "__main__":
    main()
