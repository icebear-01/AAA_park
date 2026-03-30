import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for path in [ROOT_DIR, CURRENT_DIR]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.export_vector_case_plots import (
    PALETTE,
    build_agent,
    create_env,
    draw_case,
    generate_scene_map,
    map_to_dict,
    reset_env_from_map,
    seed_everything,
)
from env.vehicle import Status


PAPER_SINGLE_SUMMARY = Path(ROOT_DIR) / "log" / "eval" / "minimal_full_sac0_20260322_183643.json"
PAPER_DUAL_NC_SUMMARY = Path(ROOT_DIR) / "log" / "eval" / "dual_normal_complex_compare_20260329_233728" / "summary.json"
PAPER_DUAL_NC_DETAILS = Path(ROOT_DIR) / "log" / "eval" / "dual_normal_complex_compare_20260329_233728" / "episode_details.json"
PAPER_DUAL_EXT_SUMMARY = Path(ROOT_DIR) / "log" / "eval" / "bidirectional_extrem_parallel_stats_20260329_221830" / "summary.json"
PAPER_DUAL_EXT_DETAILS = Path(ROOT_DIR) / "log" / "eval" / "bidirectional_extrem_parallel_stats_20260329_221830" / "episode_details.json"

METHOD_COLORS = {
    "single_final": "#355070",
    "dual_planning": "#D98F2B",
    "dual_final": "#2A9D8F",
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def scene_title(scene: str) -> str:
    return {"Normal": "Simple", "Complex": "Complex", "Extrem": "Extreme"}.get(scene, scene)


def parking_type_title(case_id: int) -> str:
    return "Bay" if case_id == 0 else "Parallel"


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "axes.labelcolor": PALETTE["dark"],
            "axes.edgecolor": PALETTE["dark"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.color": PALETTE["dark"],
            "ytick.color": PALETTE["dark"],
            "legend.fontsize": 10.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, y_grid=True):
    ax.set_facecolor("#FBFBF8")
    if y_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, color="#C9CED6")
    ax.tick_params(axis="both", length=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["dark"])
        ax.spines[spine].set_linewidth(0.8)


class SingleRunner:
    def __init__(self, ckpt_path: str):
        self.env = create_env()
        self.agent = build_agent(ckpt_path, self.env)

    def run(self, map_obj, action_seed: int, capture_trace: bool = True):
        seed_everything(action_seed)
        self.env.action_space.seed(action_seed)
        obs = reset_env_from_map(self.env, map_obj)
        self.agent.reset()

        done = False
        step_num = 0
        rs_assist_used = False
        segments = []
        while not done:
            step_num += 1
            phase = "forward_rs_assist" if self.agent.executing_rs else "forward_policy"
            prev_state = deepcopy(self.env.unwrapped.vehicle.state)
            action, _ = self.agent.get_action(obs)
            obs, reward, done, info = self.env.step(action)
            curr_state = self.env.unwrapped.vehicle.state
            if capture_trace:
                segments.append(
                    {
                        "phase": phase,
                        "x0": float(prev_state.loc.x),
                        "y0": float(prev_state.loc.y),
                        "x1": float(curr_state.loc.x),
                        "y1": float(curr_state.loc.y),
                        "action": [float(action[0]), float(action[1])],
                    }
                )
            if info["path_to_dest"] is not None:
                rs_assist_used = True
                self.agent.set_planner_path(info["path_to_dest"])

        result = {
            "method": "single",
            "status": info["status"].name,
            "final_success": bool(info["status"] == Status.ARRIVED),
            "planning_success": bool(info["status"] == Status.ARRIVED),
            "step_num": step_num,
            "rs_assist_used": rs_assist_used,
        }
        if capture_trace:
            result["trajectory_states"] = [
                {
                    "x": float(state.loc.x),
                    "y": float(state.loc.y),
                    "heading": float(state.heading),
                    "speed": float(getattr(state, "speed", 0.0)),
                    "steering": float(getattr(state, "steering", 0.0)),
                }
                for state in self.env.unwrapped.vehicle.trajectory
            ]
            result["segments"] = segments
            result["map"] = map_to_dict(self.env.unwrapped.map)
        return result

    def close(self):
        self.env.close()


class DualRunner:
    def __init__(self, forward_ckpt: str, unpark_ckpt: str):
        self.env = create_env()
        forward_agent = build_agent(forward_ckpt, self.env)
        unpark_agent = build_agent(unpark_ckpt, self.env)
        from model.agent.bidirectional_parking_agent import BidirectionalParkingAgent
        self.agent = BidirectionalParkingAgent(forward_agent, unpark_agent)

    def run(self, map_obj, action_seed: int, capture_trace: bool = True):
        seed_everything(action_seed)
        self.env.action_space.seed(action_seed)
        obs = reset_env_from_map(self.env, map_obj)
        self.agent.reset(self.env)

        done = False
        step_num = 0
        connection_step = None
        segments = []
        while not done:
            step_num += 1
            prev_connection_used = self.agent.connection_used
            prev_state = deepcopy(self.env.unwrapped.vehicle.state)
            action, _ = self.agent.choose_action(obs, self.env)
            phase = self.agent.last_action_phase
            if (not prev_connection_used) and self.agent.connection_used and connection_step is None:
                connection_step = step_num
            obs, reward, done, info = self.env.step(action)
            curr_state = self.env.unwrapped.vehicle.state
            if capture_trace:
                segments.append(
                    {
                        "phase": phase,
                        "x0": float(prev_state.loc.x),
                        "y0": float(prev_state.loc.y),
                        "x1": float(curr_state.loc.x),
                        "y1": float(curr_state.loc.y),
                        "action": [float(action[0]), float(action[1])],
                    }
                )
            if info["path_to_dest"] is not None and not self.agent.connection_used:
                self.agent.forward_agent.set_planner_path(info["path_to_dest"])

        result = {
            "method": "dual",
            "status": info["status"].name,
            "planning_success": bool(self.agent.connection_used or info["status"] == Status.ARRIVED),
            "final_success": bool(info["status"] == Status.ARRIVED),
            "connection_used": bool(self.agent.connection_used),
            "connection_index": self.agent.connection_index,
            "connection_step": connection_step,
            "step_num": step_num,
        }
        if capture_trace:
            result["trajectory_states"] = [
                {
                    "x": float(state.loc.x),
                    "y": float(state.loc.y),
                    "heading": float(state.heading),
                    "speed": float(getattr(state, "speed", 0.0)),
                    "steering": float(getattr(state, "steering", 0.0)),
                }
                for state in self.env.unwrapped.vehicle.trajectory
            ]
            result["segments"] = segments
            result["map"] = map_to_dict(self.env.unwrapped.map)
            result["reverse_states"] = [
                {
                    "x": float(state.loc.x),
                    "y": float(state.loc.y),
                    "heading": float(state.heading),
                    "speed": float(getattr(state, "speed", 0.0)),
                    "steering": float(getattr(state, "steering", 0.0)),
                }
                for state in self.agent.reverse_states
            ]
            result["connector_path"] = None if self.agent.connection_path is None else {
                "x": [float(x) for x in self.agent.connection_path.x],
                "y": [float(y) for y in self.agent.connection_path.y],
            }
        return result

    def close(self):
        self.env.close()


def collect_overall_metrics():
    single_summary = load_json(PAPER_SINGLE_SUMMARY)
    dual_nc_summary = load_json(PAPER_DUAL_NC_SUMMARY)
    dual_nc_details = load_json(PAPER_DUAL_NC_DETAILS)
    dual_ext_details = load_json(PAPER_DUAL_EXT_DETAILS)

    metrics = {}
    for scene in ("Normal", "Complex"):
        metrics[scene] = {
            "single_final_rate": float(single_summary["levels"][scene]["success_rate"]),
            "dual_planning_rate": float(dual_nc_summary["results"][scene]["path_planning_success_rate"]),
            "dual_final_rate": float(dual_nc_summary["results"][scene]["final_parking_success_rate"]),
            "breakdown": {
                "direct_park": sum(int((not item["connection_success"]) and item["final_success"]) for item in dual_nc_details[scene]),
                "connected_parked": sum(int(item["connection_success"] and item["final_success"]) for item in dual_nc_details[scene]),
                "connected_exec_fail": sum(int(item["connection_success"] and (not item["final_success"])) for item in dual_nc_details[scene]),
                "fail": sum(int((not item["connection_success"]) and (not item["final_success"])) for item in dual_nc_details[scene]),
            },
        }

    total = len(dual_ext_details)
    direct_park = sum(int((not item["connection_used"]) and item["final_parking_success"]) for item in dual_ext_details)
    connected_parked = sum(int(item["connection_used"] and item["final_parking_success"]) for item in dual_ext_details)
    connected_exec_fail = sum(int(item["connection_used"] and (not item["final_parking_success"])) for item in dual_ext_details)
    fail = sum(int((not item["connection_used"]) and (not item["final_parking_success"])) for item in dual_ext_details)
    metrics["Extrem"] = {
        "single_final_rate": float(single_summary["levels"]["Extrem"]["success_rate"]),
        "dual_planning_rate": float((direct_park + connected_parked + connected_exec_fail) / total),
        "dual_final_rate": float((direct_park + connected_parked) / total),
        "breakdown": {
            "direct_park": direct_park,
            "connected_parked": connected_parked,
            "connected_exec_fail": connected_exec_fail,
            "fail": fail,
        },
    }
    return metrics


def eval_parking_types(forward_ckpt, unpark_ckpt, seeds_per_type):
    single_runner = SingleRunner(forward_ckpt)
    dual_runner = DualRunner(forward_ckpt, unpark_ckpt)
    results = {}
    try:
        for level in ("Normal", "Complex"):
            results[level] = {}
            for case_id in (0, 1):
                label = parking_type_title(case_id)
                rows = []
                for seed in range(1, seeds_per_type + 1):
                    map_obj = generate_scene_map(level, seed, case_id)
                    action_seed = 5000 + case_id * 1000 + seed
                    single = single_runner.run(map_obj, action_seed, capture_trace=False)
                    dual = dual_runner.run(map_obj, action_seed, capture_trace=False)
                    rows.append(
                        {
                            "seed": seed,
                            "single_final_success": single["final_success"],
                            "dual_planning_success": dual["planning_success"],
                            "dual_final_success": dual["final_success"],
                            "single_steps": single["step_num"],
                            "dual_steps": dual["step_num"],
                        }
                    )
                results[level][label] = {
                    "seed_count": seeds_per_type,
                    "single_final_rate": float(np.mean([row["single_final_success"] for row in rows])),
                    "dual_planning_rate": float(np.mean([row["dual_planning_success"] for row in rows])),
                    "dual_final_rate": float(np.mean([row["dual_final_success"] for row in rows])),
                    "avg_single_steps": float(np.mean([row["single_steps"] for row in rows])),
                    "avg_dual_steps": float(np.mean([row["dual_steps"] for row in rows])),
                    "details": rows,
                }
    finally:
        single_runner.close()
        dual_runner.close()
    return results


def scan_extrem_rescues(forward_ckpt, unpark_ckpt, max_seed):
    single_runner = SingleRunner(forward_ckpt)
    dual_runner = DualRunner(forward_ckpt, unpark_ckpt)
    rows = []
    try:
        for seed in range(1, max_seed + 1):
            map_obj = generate_scene_map("Extrem", seed, 0)
            action_seed = 9000 + seed
            single = single_runner.run(map_obj, action_seed, capture_trace=False)
            dual = dual_runner.run(map_obj, action_seed, capture_trace=False)
            rows.append(
                {
                    "seed": seed,
                    "single_status": single["status"],
                    "single_success": single["final_success"],
                    "single_steps": single["step_num"],
                    "dual_status": dual["status"],
                    "dual_success": dual["final_success"],
                    "dual_planning_success": dual["planning_success"],
                    "dual_steps": dual["step_num"],
                    "connection_used": dual["connection_used"],
                    "connection_index": dual["connection_index"],
                    "step_gain": single["step_num"] - dual["step_num"],
                }
            )
    finally:
        single_runner.close()
        dual_runner.close()

    rescues = [row for row in rows if (not row["single_success"]) and row["dual_success"]]
    rescues = sorted(rescues, key=lambda row: (-row["step_gain"], row["seed"]))
    big_gains = [row for row in rows if row["single_success"] and row["dual_success"] and row["step_gain"] >= 20]
    big_gains = sorted(big_gains, key=lambda row: (-row["step_gain"], row["seed"]))
    regressions = [row for row in rows if row["single_success"] and (not row["dual_success"])]
    regressions = sorted(regressions, key=lambda row: (row["seed"]))
    return {
        "max_seed": max_seed,
        "rescues": rescues,
        "big_gains": big_gains,
        "regressions": regressions,
        "all": rows,
    }


def capture_case(level, seed, case_id, forward_ckpt, unpark_ckpt):
    map_obj = generate_scene_map(level, seed, case_id)
    action_seed = 12000 + case_id * 1000 + seed
    single_runner = SingleRunner(forward_ckpt)
    dual_runner = DualRunner(forward_ckpt, unpark_ckpt)
    try:
        single = single_runner.run(map_obj, action_seed, capture_trace=True)
        dual = dual_runner.run(map_obj, action_seed, capture_trace=True)
    finally:
        single_runner.close()
        dual_runner.close()
    return {
        "label": f"{scene_title(level)} seed {seed}",
        "level": level,
        "seed": seed,
        "case_id": case_id,
        "single": single,
        "dual": dual,
    }


def save_figure(fig, out_path: Path, png_dpi):
    fig.savefig(out_path.with_suffix(".png"), dpi=png_dpi)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_difficulty_comparison(metrics, out_dir, png_dpi):
    setup_style()
    scenes = ["Normal", "Complex", "Extrem"]
    x = np.arange(len(scenes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.3, 5.6))
    style_axis(ax)

    single_vals = [metrics[scene]["single_final_rate"] * 100 for scene in scenes]
    dual_plan_vals = [metrics[scene]["dual_planning_rate"] * 100 for scene in scenes]
    dual_final_vals = [metrics[scene]["dual_final_rate"] * 100 for scene in scenes]

    bars = []
    bars.append(ax.bar(x - width, single_vals, width, color=METHOD_COLORS["single_final"], edgecolor="white", linewidth=0.8, label="HOPE final"))
    bars.append(ax.bar(x, dual_plan_vals, width, color=METHOD_COLORS["dual_planning"], edgecolor="white", linewidth=0.8, label="Dual planning"))
    bars.append(ax.bar(x + width, dual_final_vals, width, color=METHOD_COLORS["dual_final"], edgecolor="white", linewidth=0.8, label="Dual final"))
    ax.set_ylim(70, 101.5)
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([scene_title(scene) for scene in scenes])
    ax.set_title("Success Rate Across Difficulty Levels")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    for group in bars:
        for bar in group:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3, f"{height:.1f}", ha="center", va="bottom", fontsize=9.3, color=PALETTE["dark"])
    save_figure(fig, out_dir / "fig01_difficulty_success", png_dpi)


def plot_parking_type_comparison(parking_type_results, out_dir, png_dpi):
    setup_style()
    groups = [("Normal", "Bay"), ("Normal", "Parallel"), ("Complex", "Bay"), ("Complex", "Parallel")]
    x = np.arange(len(groups))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    style_axis(ax)

    single_vals = [parking_type_results[level][ptype]["single_final_rate"] * 100 for level, ptype in groups]
    dual_plan_vals = [parking_type_results[level][ptype]["dual_planning_rate"] * 100 for level, ptype in groups]
    dual_final_vals = [parking_type_results[level][ptype]["dual_final_rate"] * 100 for level, ptype in groups]

    bars = []
    bars.append(ax.bar(x - width, single_vals, width, color=METHOD_COLORS["single_final"], edgecolor="white", linewidth=0.8, label="HOPE final"))
    bars.append(ax.bar(x, dual_plan_vals, width, color=METHOD_COLORS["dual_planning"], edgecolor="white", linewidth=0.8, label="Dual planning"))
    bars.append(ax.bar(x + width, dual_final_vals, width, color=METHOD_COLORS["dual_final"], edgecolor="white", linewidth=0.8, label="Dual final"))

    ax.set_ylim(65, 101.5)
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{scene_title(level)}\n{ptype}" for level, ptype in groups])
    ax.set_title("Success Rate Across Parking Slot Types")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    for group in bars:
        for bar in group:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3, f"{height:.1f}", ha="center", va="bottom", fontsize=8.9, color=PALETTE["dark"])
    ax.text(1.0, -0.18, "Extrem currently uses parallel-slot generation only in the released map code.", transform=ax.transAxes, ha="right", va="top", fontsize=9.0, color=PALETTE["dark"], alpha=0.8)
    save_figure(fig, out_dir / "fig02_parking_type_success", png_dpi)


def plot_dual_breakdown(metrics, out_dir, png_dpi):
    setup_style()
    scenes = ["Normal", "Complex", "Extrem"]
    categories = [
        ("direct_park", "Direct park", "#5FA8D3"),
        ("connected_parked", "Connected + parked", "#2A9D8F"),
        ("connected_exec_fail", "Connected + execution fail", "#E9C46A"),
        ("fail", "No path / fail", "#C8553D"),
    ]
    totals = [sum(metrics[scene]["breakdown"].values()) for scene in scenes]
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    style_axis(ax)
    bottom = np.zeros(len(scenes))
    x = np.arange(len(scenes))
    for key, label, color in categories:
        values = [metrics[scene]["breakdown"][key] / totals[idx] * 100 for idx, scene in enumerate(scenes)]
        ax.bar(x, values, bottom=bottom, color=color, edgecolor="white", linewidth=0.8, label=label)
        bottom += np.array(values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Ratio within Dual-Model Outcomes (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([scene_title(scene) for scene in scenes])
    ax.set_title("Dual-Model Outcome Decomposition")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save_figure(fig, out_dir / "fig03_dual_breakdown", png_dpi)


def plot_case_panel(cases, out_dir, stem, title, png_dpi):
    setup_style()
    fig, axes = plt.subplots(nrows=len(cases), ncols=2, figsize=(14.0, 4.9 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])
    for idx, case in enumerate(cases):
        draw_case(axes[idx, 0], case["label"], case["single"], show_legend=(idx == 0), show_axis_labels=(idx == len(cases) - 1))
        draw_case(axes[idx, 1], case["label"], case["dual"], show_legend=False, show_axis_labels=(idx == len(cases) - 1))
    fig.suptitle(title, fontsize=15, y=0.996, color=PALETTE["dark"])
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save_figure(fig, out_dir / stem, png_dpi)


def plot_failure_and_action_heatmaps(case, out_dir, png_dpi):
    setup_style()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5.3))

    single = case["single"]
    dual = case["dual"]
    draw_case(axes[0], case["label"], single, show_legend=False, show_axis_labels=True)
    xs = [state["x"] for state in single["trajectory_states"]]
    ys = [state["y"] for state in single["trajectory_states"]]
    axes[0].hexbin(xs, ys, gridsize=32, cmap="YlOrRd", mincnt=1, alpha=0.55, linewidths=0.0, zorder=11)
    axes[0].set_title(f"{case['label']} | HOPE failed node density", fontsize=12.0, color=PALETTE["dark"])

    actions = np.array([segment["action"] for segment in dual["segments"]], dtype=np.float64)
    hist = axes[1].hist2d(actions[:, 0], actions[:, 1], bins=18, cmap="YlGnBu")
    axes[1].set_xlabel("Steering command")
    axes[1].set_ylabel("Speed command")
    axes[1].set_title(f"{case['label']} | Dual action density", fontsize=12.0, color=PALETTE["dark"])
    axes[1].grid(False)
    for spine in ("left", "bottom"):
        axes[1].spines[spine].set_color(PALETTE["dark"])
        axes[1].spines[spine].set_linewidth(0.8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    cbar = fig.colorbar(hist[3], ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Frequency")

    fig.suptitle("Representative Failure and Dual-Model Control Heatmaps", fontsize=15, y=0.996, color=PALETTE["dark"])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, out_dir / "fig06_heatmaps", png_dpi)


def write_notes(out_dir, metrics, parking_type_results, scan_results, selected_cases):
    notes = []
    notes.append("# Dual-Model Paper Figure Package")
    notes.append("")
    notes.append("## Suggested Main Results")
    notes.append(f"- Overall difficulty figure: use `fig01_difficulty_success`.")
    notes.append(f"- Parking slot type figure: use `fig02_parking_type_success`.")
    notes.append(f"- Dual breakdown figure: use `fig03_dual_breakdown`.")
    notes.append(f"- Rescue cases panel: use `fig04_rescue_cases`.")
    notes.append(f"- Good planning panel: use `fig05_good_plans`.")
    notes.append(f"- Heatmaps: use `fig06_heatmaps`.")
    notes.append("")
    notes.append("## Key Numbers")
    for scene in ("Normal", "Complex", "Extrem"):
        scene_metrics = metrics[scene]
        notes.append(
            f"- {scene_title(scene)}: HOPE final {scene_metrics['single_final_rate']*100:.2f}%, "
            f"Dual planning {scene_metrics['dual_planning_rate']*100:.2f}%, "
            f"Dual final {scene_metrics['dual_final_rate']*100:.2f}%."
        )
    notes.append("")
    notes.append(f"- Extrem rescue seeds found in scan: {[row['seed'] for row in scan_results['rescues'][:8]]}")
    notes.append(f"- Parking-type evaluation seeds per type: {next(iter(next(iter(parking_type_results.values())).values()))['seed_count']}")
    (out_dir / "paper_figure_notes.md").write_text("\n".join(notes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward_ckpt", type=str, default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--unpark_ckpt", type=str, default="src/log/exp/unpark_ppo_20260323_120610/PPO_unpark_best.pt")
    parser.add_argument("--parking_type_seeds", type=int, default=100)
    parser.add_argument("--scan_extrem_max_seed", type=int, default=200)
    parser.add_argument("--png_dpi", type=int, default=320)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = Path(args.save_dir) if args.save_dir else Path(ROOT_DIR) / "log" / "paper_support" / f"dual_framework_{timestamp}"
    ensure_dir(save_dir)

    print("[1/5] collect overall metrics")
    metrics = collect_overall_metrics()
    (save_dir / "overall_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("[2/5] evaluate parking slot types")
    parking_type_results = eval_parking_types(args.forward_ckpt, args.unpark_ckpt, args.parking_type_seeds)
    (save_dir / "parking_type_results.json").write_text(json.dumps(parking_type_results, indent=2))

    print("[3/5] scan Extrem rescue cases")
    scan_results = scan_extrem_rescues(args.forward_ckpt, args.unpark_ckpt, args.scan_extrem_max_seed)
    (save_dir / "extrem_scan_results.json").write_text(json.dumps(scan_results, indent=2))

    rescue_specs = [(row["seed"], 0) for row in scan_results["rescues"][:3]]
    if len(rescue_specs) < 3:
        rescue_specs.extend([(row["seed"], 0) for row in scan_results["big_gains"][: 3 - len(rescue_specs)]])
    good_specs = [(5, 0, "Complex"), (40, 0, "Complex"), (31, 0, "Extrem"), (37, 0, "Extrem")]

    print("[4/5] capture selected qualitative cases")
    rescue_cases = [capture_case("Extrem", seed, case_id, args.forward_ckpt, args.unpark_ckpt) for seed, case_id in rescue_specs]
    good_cases = [capture_case(level, seed, case_id, args.forward_ckpt, args.unpark_ckpt) for seed, case_id, level in good_specs]

    selected = {
        "rescue_cases": [{"level": case["level"], "seed": case["seed"], "case_id": case["case_id"]} for case in rescue_cases],
        "good_cases": [{"level": case["level"], "seed": case["seed"], "case_id": case["case_id"]} for case in good_cases],
    }
    (save_dir / "selected_cases.json").write_text(json.dumps(selected, indent=2))

    print("[5/5] render paper figures")
    plot_difficulty_comparison(metrics, save_dir, args.png_dpi)
    plot_parking_type_comparison(parking_type_results, save_dir, args.png_dpi)
    plot_dual_breakdown(metrics, save_dir, args.png_dpi)
    plot_case_panel(rescue_cases, save_dir, "fig04_rescue_cases", "Representative Rescue Cases: HOPE Fails, Dual Model Succeeds", args.png_dpi)
    plot_case_panel(good_cases, save_dir, "fig05_good_plans", "Representative Cases with Shorter or Cleaner Dual-Model Plans", args.png_dpi)
    if rescue_cases:
        plot_failure_and_action_heatmaps(rescue_cases[0], save_dir, args.png_dpi)

    write_notes(save_dir, metrics, parking_type_results, scan_results, selected)

    manifest = {
        "save_dir": str(save_dir),
        "figures": [
            "fig01_difficulty_success",
            "fig02_parking_type_success",
            "fig03_dual_breakdown",
            "fig04_rescue_cases",
            "fig05_good_plans",
            "fig06_heatmaps",
        ],
        "rescue_seed_examples": [case["seed"] for case in rescue_cases],
    }
    (save_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
