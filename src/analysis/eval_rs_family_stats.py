import argparse
import json
import math
import os
import sys
import time
from collections import Counter
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
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_rs_family_stats")

import env.reeds_shepp as rs_curve
from configs import VALID_STEER, WHEEL_BASE
from env.vehicle import Status
from model.agent.bidirectional_parking_agent import BidirectionalParkingAgent
from analysis.export_vector_case_plots import (
    build_agent,
    create_env,
    generate_scene_map,
    reset_env_from_map,
    seed_everything,
)


SCENE_SPECS = [
    {"level": "Normal", "case_id": 0, "tag": "normal_bay", "name": "Normal Bay"},
    {"level": "Complex", "case_id": 0, "tag": "complex_bay", "name": "Complex Bay"},
    {"level": "Normal", "case_id": 1, "tag": "normal_parallel", "name": "Normal Parallel"},
    {"level": "Complex", "case_id": 1, "tag": "complex_parallel", "name": "Complex Parallel"},
    {"level": "Extrem", "case_id": 1, "tag": "extrem_parallel", "name": "Extrem Parallel"},
]


def family_name(ctypes):
    return "".join("C" if char in {"L", "R"} else char for char in ctypes)


def safe_rate(count, total):
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def counter_to_ranked_dict(counter):
    return {
        key: int(value)
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    }


def reverse_state_to_dict(state):
    return {
        "x": float(state.loc.x),
        "y": float(state.loc.y),
        "heading": float(state.heading),
        "speed": float(getattr(state, "speed", 0.0)),
        "steering": float(getattr(state, "steering", 0.0)),
    }


def collect_feasible_family_info(env, start_state, goal_state):
    radius = math.tan(VALID_STEER[-1]) / WHEEL_BASE
    all_paths = rs_curve.calc_all_paths(
        start_state.loc.x,
        start_state.loc.y,
        start_state.heading,
        goal_state.loc.x,
        goal_state.loc.y,
        goal_state.heading,
        radius,
        0.1,
    )
    all_paths = sorted(all_paths, key=lambda path: path.L)

    feasible = {}
    for path in all_paths:
        family = family_name(path.ctypes)
        if family in feasible:
            continue
        traj = [[path.x[idx], path.y[idx], path.yaw[idx]] for idx in range(len(path.x))]
        if not env.unwrapped.is_traj_valid(traj):
            continue
        feasible[family] = {
            "label": rs_curve.get_label(path),
            "length": float(path.L),
        }
    return feasible


class DualFamilyRunner:
    def __init__(self, forward_ckpt, unpark_ckpt):
        self.env = create_env()
        forward_agent = build_agent(forward_ckpt, self.env)
        unpark_agent = build_agent(unpark_ckpt, self.env)
        self.agent = BidirectionalParkingAgent(forward_agent, unpark_agent)

    def run_episode(self, map_obj, action_seed):
        seed_everything(action_seed)
        self.env.action_space.seed(action_seed)
        obs = reset_env_from_map(self.env, map_obj)
        self.agent.reset(self.env)

        done = False
        step_num = 0
        connection_start_state = None

        while not done:
            step_num += 1
            prev_connection_used = self.agent.connection_used
            prev_state = deepcopy(self.env.unwrapped.vehicle.state)
            action, _ = self.agent.choose_action(obs, self.env)
            if (not prev_connection_used) and self.agent.connection_used and connection_start_state is None:
                connection_start_state = deepcopy(prev_state)
            obs, reward, done, info = self.env.step(action)

        selected_family = None
        selected_label = None
        feasible_family_info = {}
        connection_goal_state = None
        if self.agent.connection_used and self.agent.connection_path is not None:
            selected_family = family_name(self.agent.connection_path.ctypes)
            selected_label = rs_curve.get_label(self.agent.connection_path)
            if connection_start_state is not None and self.agent.connection_index is not None:
                connection_goal_state = deepcopy(self.agent.reverse_states[self.agent.connection_index])
                feasible_family_info = collect_feasible_family_info(
                    self.env,
                    connection_start_state,
                    connection_goal_state,
                )

        return {
            "status": info["status"].name,
            "planning_success": bool(self.agent.connection_used or info["status"] == Status.ARRIVED),
            "final_success": bool(info["status"] == Status.ARRIVED),
            "connection_used": bool(self.agent.connection_used),
            "connection_index": self.agent.connection_index,
            "step_num": int(step_num),
            "selected_family": selected_family,
            "selected_label": selected_label,
            "feasible_families_at_connection": sorted(feasible_family_info.keys()),
            "feasible_family_info": feasible_family_info,
            "connection_goal_state": None if connection_goal_state is None else reverse_state_to_dict(connection_goal_state),
        }

    def close(self):
        self.env.close()


def default_action_seed(case_id, seed):
    return 61000 + case_id * 1000 + seed


def summarize_scene(rows):
    total = len(rows)
    planning_success = sum(int(row["planning_success"]) for row in rows)
    final_success = sum(int(row["final_success"]) for row in rows)
    connection_used = sum(int(row["connection_used"]) for row in rows)

    selected_counts = Counter()
    selected_final_success_counts = Counter()
    feasible_presence_counts = Counter()
    feasible_final_success_counts = Counter()
    scs_selected_seeds = []
    scs_feasible_seeds = []

    for row in rows:
        family = row["selected_family"]
        if family is not None:
            selected_counts[family] += 1
            if row["final_success"]:
                selected_final_success_counts[family] += 1
            if family == "SCS":
                scs_selected_seeds.append(int(row["seed"]))

        feasible_families = set(row["feasible_families_at_connection"])
        for family in feasible_families:
            feasible_presence_counts[family] += 1
            if row["final_success"]:
                feasible_final_success_counts[family] += 1
        if "SCS" in feasible_families:
            scs_feasible_seeds.append(int(row["seed"]))

    family_stats = {}
    all_families = sorted(set(selected_counts.keys()) | set(feasible_presence_counts.keys()))
    for family in all_families:
        sel_count = int(selected_counts.get(family, 0))
        feas_count = int(feasible_presence_counts.get(family, 0))
        family_stats[family] = {
            "selected_count": sel_count,
            "selected_rate_among_all": safe_rate(sel_count, total),
            "selected_rate_among_connections": safe_rate(sel_count, connection_used),
            "selected_final_success_count": int(selected_final_success_counts.get(family, 0)),
            "selected_final_success_rate": safe_rate(selected_final_success_counts.get(family, 0), sel_count),
            "feasible_presence_count": feas_count,
            "feasible_presence_rate_among_all": safe_rate(feas_count, total),
            "feasible_presence_rate_among_connections": safe_rate(feas_count, connection_used),
            "feasible_final_success_count": int(feasible_final_success_counts.get(family, 0)),
            "feasible_final_success_rate": safe_rate(feasible_final_success_counts.get(family, 0), feas_count),
        }

    return {
        "episodes": total,
        "planning_success_count": planning_success,
        "planning_success_rate": safe_rate(planning_success, total),
        "final_success_count": final_success,
        "final_success_rate": safe_rate(final_success, total),
        "connection_used_count": connection_used,
        "connection_used_rate": safe_rate(connection_used, total),
        "selected_family_counts": counter_to_ranked_dict(selected_counts),
        "selected_family_final_success_counts": counter_to_ranked_dict(selected_final_success_counts),
        "feasible_family_presence_counts": counter_to_ranked_dict(feasible_presence_counts),
        "feasible_family_final_success_counts": counter_to_ranked_dict(feasible_final_success_counts),
        "family_stats": family_stats,
        "scs_selected_seeds": scs_selected_seeds,
        "scs_feasible_seeds": scs_feasible_seeds,
    }


def print_scene_summary(scene_name, scene_summary):
    print(
        f"{scene_name}: episodes={scene_summary['episodes']}, "
        f"conn={scene_summary['connection_used_count']} "
        f"({scene_summary['connection_used_rate'] * 100:.2f}%), "
        f"plan={scene_summary['planning_success_rate'] * 100:.2f}%, "
        f"final={scene_summary['final_success_rate'] * 100:.2f}%"
    )
    for family, stats in sorted(
        scene_summary["family_stats"].items(),
        key=lambda item: (-item[1]["selected_count"], item[0]),
    ):
        print(
            f"  {family}: "
            f"selected={stats['selected_count']} "
            f"({stats['selected_rate_among_connections'] * 100:.2f}% of connections), "
            f"feasible={stats['feasible_presence_count']} "
            f"({stats['feasible_presence_rate_among_connections'] * 100:.2f}% of connections)"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate RS family usage statistics for the dual parking model.")
    parser.add_argument("--forward_ckpt", type=str, default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--unpark_ckpt", type=str, default="src/log/exp/unpark_ppo_20260323_120610/PPO_unpark_best.pt")
    parser.add_argument("--seeds_per_spec", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(ROOT_DIR) / "log" / "analysis" / f"rs_family_stats_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = DualFamilyRunner(args.forward_ckpt, args.unpark_ckpt)
    try:
        all_rows = []
        per_scene = {}
        for idx, spec in enumerate(SCENE_SPECS, start=1):
            print(f"[{idx}/{len(SCENE_SPECS)}] {spec['name']}")
            rows = []
            for seed in range(1, args.seeds_per_spec + 1):
                map_obj = generate_scene_map(spec["level"], seed, spec["case_id"])
                action_seed = default_action_seed(spec["case_id"], seed)
                episode = runner.run_episode(map_obj, action_seed)
                episode["seed"] = int(seed)
                episode["action_seed"] = int(action_seed)
                episode["level"] = spec["level"]
                episode["case_id"] = int(spec["case_id"])
                episode["tag"] = spec["tag"]
                episode["scene_name"] = spec["name"]
                rows.append(episode)
                all_rows.append(episode)
            scene_summary = summarize_scene(rows)
            per_scene[spec["tag"]] = {
                "name": spec["name"],
                "level": spec["level"],
                "case_id": int(spec["case_id"]),
                "summary": scene_summary,
                "episodes": rows,
            }
            print_scene_summary(spec["name"], scene_summary)

        overall_summary = summarize_scene(all_rows)
        print_scene_summary("Overall", overall_summary)

        output = {
            "forward_ckpt": args.forward_ckpt,
            "unpark_ckpt": args.unpark_ckpt,
            "seeds_per_spec": int(args.seeds_per_spec),
            "scene_specs": SCENE_SPECS,
            "overall_summary": overall_summary,
            "scenes": per_scene,
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(summary_path)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
