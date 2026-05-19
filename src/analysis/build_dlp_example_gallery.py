import argparse
import json
import math
import os
import sys
import time
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

import export_dual_parking_paper_gallery as paper_gallery
from analysis.export_vector_case_plots import (
    build_agent,
    create_env,
    ensure_dir,
    generate_scene_map,
    map_to_dict,
    reset_env_from_map,
    seed_everything,
)
from env.vehicle import Status
from model.agent.bidirectional_parking_agent import BidirectionalParkingAgent

DEFAULT_FOCUS_WIDTH = 36.0
DEFAULT_FOCUS_HEIGHT = 32.0
DEFAULT_FOCUS_SNAP = 2.0


def sign_of_speed(action):
    speed = float(action[1])
    if speed > 1e-6:
        return 1
    if speed < -1e-6:
        return -1
    return 0


def extract_motion_runs(segments):
    runs = []
    prev_sign = 0
    current = None
    for seg in segments:
        sign = sign_of_speed(seg["action"])
        if sign == 0:
            continue
        dist = float(np.hypot(seg["x1"] - seg["x0"], seg["y1"] - seg["y0"]))
        if prev_sign == 0 or sign != prev_sign:
            current = {"sign": sign, "steps": 1, "distance": dist}
            runs.append(current)
        else:
            current["steps"] += 1
            current["distance"] += dist
        prev_sign = sign
    return runs


def count_motion_segments(segments, min_run_steps=2, min_run_distance=0.9):
    runs = extract_motion_runs(segments)
    if not runs:
        return 0, 0

    filtered = []
    for run in runs:
        if run["steps"] < min_run_steps and run["distance"] < min_run_distance:
            continue
        filtered.append(dict(run))

    if not filtered:
        dominant = max(runs, key=lambda item: (item["distance"], item["steps"]))
        filtered = [dict(dominant)]

    merged = []
    for run in filtered:
        if merged and merged[-1]["sign"] == run["sign"]:
            merged[-1]["steps"] += run["steps"]
            merged[-1]["distance"] += run["distance"]
        else:
            merged.append(dict(run))
    return len(merged), len(runs)


def default_action_seed(case_id, scene_seed):
    return 51000 + int(case_id) * 100 + int(scene_seed)


class DualCaseRunner:
    def __init__(self, forward_ckpt: str, unpark_ckpt: str):
        self.env = create_env()
        self.forward_agent = build_agent(forward_ckpt, self.env)
        self.unpark_agent = build_agent(unpark_ckpt, self.env)
        self.agent = BidirectionalParkingAgent(self.forward_agent, self.unpark_agent)

    def run(self, map_obj, action_seed: int):
        seed_everything(action_seed)
        self.env.action_space.seed(action_seed)
        obs = reset_env_from_map(self.env, map_obj)

        prepare_t0 = time.perf_counter()
        self.agent.reset(self.env)
        prepare_elapsed_ms = (time.perf_counter() - prepare_t0) * 1000.0

        done = False
        step_num = 0
        connection_step = None
        segments = []
        phase_counts = {}
        rollout_t0 = time.perf_counter()

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
            phase_counts.setdefault(phase, 0)
            phase_counts[phase] += 1
            if info["path_to_dest"] is not None and not self.agent.connection_used:
                self.agent.forward_agent.set_planner_path(info["path_to_dest"])

        rollout_elapsed_ms = (time.perf_counter() - rollout_t0) * 1000.0
        total_elapsed_ms = prepare_elapsed_ms + rollout_elapsed_ms

        return {
            "method": "dual",
            "status": info["status"].name,
            "final_success": bool(info["status"] == Status.ARRIVED),
            "planning_success": bool(self.agent.connection_used or info["status"] == Status.ARRIVED),
            "connection_used": bool(self.agent.connection_used),
            "connection_index": self.agent.connection_index,
            "connection_step": connection_step,
            "step_num": step_num,
            "inference_time_ms": float(total_elapsed_ms),
            "prepare_time_ms": float(prepare_elapsed_ms),
            "rollout_time_ms": float(rollout_elapsed_ms),
            "avg_step_time_ms": float(total_elapsed_ms / max(step_num, 1)),
            "trajectory_states": [
                {
                    "x": float(state.loc.x),
                    "y": float(state.loc.y),
                    "heading": float(state.heading),
                    "speed": float(getattr(state, "speed", 0.0)),
                    "steering": float(getattr(state, "steering", 0.0)),
                }
                for state in self.env.unwrapped.vehicle.trajectory
            ],
            "segments": segments,
            "phase_steps": phase_counts,
            "reverse_states": [
                {
                    "x": float(state.loc.x),
                    "y": float(state.loc.y),
                    "heading": float(state.heading),
                    "speed": float(getattr(state, "speed", 0.0)),
                    "steering": float(getattr(state, "steering", 0.0)),
                }
                for state in self.agent.reverse_states
            ],
            "reverse_actions": [action.tolist() for action in self.agent.reverse_actions],
            "connector_path": None
            if self.agent.connection_path is None
            else {
                "x": [float(x) for x in self.agent.connection_path.x],
                "y": [float(y) for y in self.agent.connection_path.y],
                "yaw": [float(yaw) for yaw in self.agent.connection_path.yaw],
                "lengths": [float(v) for v in self.agent.connection_path.lengths],
                "ctypes": list(self.agent.connection_path.ctypes),
            },
            "map": map_to_dict(self.env.unwrapped.map),
        }


def evaluate_case(case_id, scene_seed, runner):
    map_obj = generate_scene_map("dlp", scene_seed, case_id)
    action_seed = default_action_seed(case_id, scene_seed)
    dual = runner.run(map_obj, action_seed)
    motion_segments, raw_motion_segments = count_motion_segments(dual["segments"])
    start = map_obj.start.loc
    dest = map_obj.dest.loc
    return {
        "case_id": int(case_id),
        "scene_seed": int(scene_seed),
        "action_seed": int(action_seed),
        "dual": dual,
        "map": map_to_dict(map_obj),
        "map_level": getattr(map_obj, "map_level", None),
        "obstacle_num": int(len(map_obj.obstacles)),
        "motion_segments": int(motion_segments),
        "raw_motion_segments": int(raw_motion_segments),
        "start_dest_distance": float(start.distance(dest)),
        "connection_used": bool(dual.get("connection_used", False)),
        "connection_index": dual.get("connection_index"),
    }


def qualify(row, min_segments, max_segments, min_steps, max_steps, min_start_dist):
    if not row["dual"]["final_success"]:
        return False
    if row["motion_segments"] < min_segments or row["motion_segments"] > max_segments:
        return False
    if row["dual"]["step_num"] < min_steps:
        return False
    if row["dual"]["step_num"] > max_steps:
        return False
    if row["start_dest_distance"] < min_start_dist:
        return False
    return True


def candidate_score(row):
    # Prefer readable 2-3 segment successful cases with moderate step counts
    # and a non-trivial obstacle layout for paper figures.
    return (
        1000.0
        - 140.0 * abs(row["motion_segments"] - 2)
        - 1.0 * float(row["dual"]["step_num"])
        + 6.0 * min(float(row["obstacle_num"]), 8.0)
        + 0.1 * min(float(row["start_dest_distance"]), 40.0)
    )


def select_cases(
    case_ids,
    scene_seeds,
    runner,
    num_cases,
    min_segments,
    max_segments,
    min_steps,
    max_steps,
    min_start_dist,
):
    best_rows = {}
    scan_rows = []

    for case_id in case_ids:
        best_row = None
        for scene_seed in scene_seeds:
            row = evaluate_case(case_id, scene_seed, runner)
            row["qualified"] = qualify(
                row,
                min_segments=min_segments,
                max_segments=max_segments,
                min_steps=min_steps,
                max_steps=max_steps,
                min_start_dist=min_start_dist,
            )
            row["score"] = candidate_score(row) if row["qualified"] else None
            scan_rows.append(
                {
                    "case_id": row["case_id"],
                    "scene_seed": row["scene_seed"],
                    "action_seed": row["action_seed"],
                    "qualified": bool(row["qualified"]),
                    "score": row["score"],
                    "step_num": int(row["dual"]["step_num"]),
                    "final_success": bool(row["dual"]["final_success"]),
                    "planning_success": bool(row["dual"]["planning_success"]),
                    "motion_segments": int(row["motion_segments"]),
                    "raw_motion_segments": int(row["raw_motion_segments"]),
                    "obstacle_num": int(row["obstacle_num"]),
                    "map_level": row["map_level"],
                    "inference_time_ms": float(row["dual"].get("inference_time_ms", 0.0)),
                    "connection_used": bool(row["connection_used"]),
                }
            )
            if not row["qualified"]:
                continue
            if best_row is None or row["score"] > best_row["score"]:
                best_row = row
        if best_row is not None:
            best_rows[int(case_id)] = best_row

    selected = sorted(
        best_rows.values(),
        key=lambda row: (
            abs(row["motion_segments"] - 2),
            row["dual"]["step_num"],
            -row["score"],
            row["case_id"],
        ),
    )[:num_cases]
    return selected, scan_rows


def snap_center(value: float, snap: float):
    if snap <= 0:
        return float(value)
    return float(snap * round(float(value) / snap))


def collect_focus_points(result):
    points = []
    for state in result.get("trajectory_states", []):
        points.append((float(state["x"]), float(state["y"])))

    map_payload = result.get("map", {})
    for key in ("start_box", "dest_box"):
        for pt in map_payload.get(key, []):
            points.append((float(pt[0]), float(pt[1])))

    connector = result.get("connector_path")
    if connector is not None:
        points.extend(zip(connector.get("x", []), connector.get("y", [])))

    if not points:
        start = map_payload.get("start", {})
        dest = map_payload.get("dest", {})
        if start:
            points.append((float(start["x"]), float(start["y"])))
        if dest:
            points.append((float(dest["x"]), float(dest["y"])))
    return np.asarray(points, dtype=np.float64)


def compute_focus_bounds(result, focus_width: float, focus_height: float, focus_snap: float):
    pts = collect_focus_points(result)
    if pts.size == 0:
        center_x, center_y = 0.0, 0.0
    else:
        center_x = 0.5 * (float(np.min(pts[:, 0])) + float(np.max(pts[:, 0])))
        center_y = 0.5 * (float(np.min(pts[:, 1])) + float(np.max(pts[:, 1])))
    center_x = snap_center(center_x, focus_snap)
    center_y = snap_center(center_y, focus_snap)
    half_w = 0.5 * float(focus_width)
    half_h = 0.5 * float(focus_height)
    return {
        "xmin": float(center_x - half_w),
        "xmax": float(center_x + half_w),
        "ymin": float(center_y - half_h),
        "ymax": float(center_y + half_h),
    }


def load_rows_from_existing_output(reuse_dir: Path):
    manifest_path = reuse_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = []
    for case_summary in manifest.get("cases", []):
        name = case_summary["name"]
        trace_path = reuse_dir / "cases" / f"{name}_trace.json"
        if not trace_path.exists():
            raise FileNotFoundError(f"trace not found: {trace_path}")
        dual = json.loads(trace_path.read_text(encoding="utf-8"))
        motion_segments, raw_motion_segments = count_motion_segments(dual.get("segments", []))
        map_payload = dual.get("map", {})
        start = map_payload.get("start", {})
        dest = map_payload.get("dest", {})
        start_dest_distance = float(
            math.hypot(float(start.get("x", 0.0)) - float(dest.get("x", 0.0)),
                       float(start.get("y", 0.0)) - float(dest.get("y", 0.0)))
        )
        rows.append(
            {
                "case_id": int(case_summary["case_id"]),
                "scene_seed": int(case_summary["scene_seed"]),
                "action_seed": int(case_summary["action_seed"]),
                "dual": dual,
                "map": map_payload,
                "map_level": case_summary.get("map_level"),
                "obstacle_num": int(case_summary.get("obstacle_num", len(map_payload.get("obstacles", [])))),
                "motion_segments": int(motion_segments),
                "raw_motion_segments": int(raw_motion_segments),
                "start_dest_distance": float(start_dest_distance),
                "connection_used": bool(dual.get("connection_used", False)),
                "connection_index": dual.get("connection_index"),
                "score": float(case_summary.get("score", 0.0)),
            }
        )
    return rows, manifest


def render_single(row, plot_bounds, out_dir: Path, png_dpi: int, min_stride: int, target_footprints: int):
    name = f"dlp_case{row['case_id']:03d}_seed{row['scene_seed']}"
    result_for_plot = paper_gallery.clone_result_with_bounds(row["dual"], plot_bounds)
    stride = paper_gallery.choose_stride(
        len(row["dual"]["trajectory_states"]),
        min_stride=min_stride,
        target_footprints=target_footprints,
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.7))
    paper_gallery.vector_plots.draw_case(
        ax,
        "",
        result_for_plot,
        show_legend=False,
        show_axis_labels=True,
        trajectory_style="footprints",
        footprint_stride=stride,
    )
    ax.set_title("")
    ax.set_xlabel("x (m)", fontsize=14.0, color=paper_gallery.vector_plots.PALETTE["dark"])
    ax.set_ylabel("y (m)", fontsize=14.0, color=paper_gallery.vector_plots.PALETTE["dark"])
    ax.tick_params(labelsize=19.0)

    base = out_dir / name
    fig.savefig(base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "name": name,
        "case_id": int(row["case_id"]),
        "scene_seed": int(row["scene_seed"]),
        "action_seed": int(row["action_seed"]),
        "map_level": row["map_level"],
        "obstacle_num": int(row["obstacle_num"]),
        "motion_segments": int(row["motion_segments"]),
        "raw_motion_segments": int(row["raw_motion_segments"]),
        "step_num": int(row["dual"]["step_num"]),
        "planning_success": bool(row["dual"]["planning_success"]),
        "final_success": bool(row["dual"]["final_success"]),
        "connection_used": bool(row["connection_used"]),
        "connection_index": row["connection_index"],
        "start_dest_distance": float(row["start_dest_distance"]),
        "inference_time_ms": float(row["dual"].get("inference_time_ms", 0.0)),
        "prepare_time_ms": float(row["dual"].get("prepare_time_ms", 0.0)),
        "rollout_time_ms": float(row["dual"].get("rollout_time_ms", 0.0)),
        "avg_step_time_ms": float(row["dual"].get("avg_step_time_ms", 0.0)),
        "stride": int(stride),
        "score": float(row["score"]),
        "plot_bounds": {key: float(value) for key, value in plot_bounds.items()},
    }
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"{name}_trace.json").write_text(json.dumps(row["dual"], indent=2), encoding="utf-8")
    return summary


def render_panel(rows, out_dir: Path, png_dpi: int, min_stride: int, target_footprints: int, panel_cols: int, focus_width: float, focus_height: float, focus_snap: float):
    n = len(rows)
    ncols = max(1, min(int(panel_cols), n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.0 * ncols, 5.8 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    manifest_rows = []
    for idx, row in enumerate(rows):
        ax = axes[idx // ncols, idx % ncols]
        plot_bounds = compute_focus_bounds(row["dual"], focus_width, focus_height, focus_snap)
        result_for_plot = paper_gallery.clone_result_with_bounds(row["dual"], plot_bounds)
        stride = paper_gallery.choose_stride(
            len(row["dual"]["trajectory_states"]),
            min_stride=min_stride,
            target_footprints=target_footprints,
        )
        paper_gallery.vector_plots.draw_case(
            ax,
            "",
            result_for_plot,
            show_legend=False,
            show_axis_labels=True,
            trajectory_style="footprints",
            footprint_stride=stride,
        )
        ax.set_title("")
        ax.set_xlabel("x (m)", fontsize=14.0, color=paper_gallery.vector_plots.PALETTE["dark"])
        ax.set_ylabel("y (m)", fontsize=14.0, color=paper_gallery.vector_plots.PALETTE["dark"])
        ax.tick_params(labelsize=18.0)
        manifest_rows.append(
            {
                "case_id": int(row["case_id"]),
                "scene_seed": int(row["scene_seed"]),
                "action_seed": int(row["action_seed"]),
                "stride": int(stride),
                "motion_segments": int(row["motion_segments"]),
                "step_num": int(row["dual"]["step_num"]),
                "inference_time_ms": float(row["dual"].get("inference_time_ms", 0.0)),
                "plot_bounds": {key: float(value) for key, value in plot_bounds.items()},
            }
        )

    for idx in range(n, nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        ax.axis("off")

    fig.tight_layout()
    base = out_dir / "dlp_examples_panel"
    fig.savefig(base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description="Select and export several readable DLP examples.")
    parser.add_argument("--forward-ckpt", type=str, default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--unpark-ckpt", type=str, default="src/log/exp/unpark_ppo_20260323_120610/PPO_unpark_best.pt")
    parser.add_argument("--scene-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--case-stop", type=int, default=248)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--min-start-dist", type=float, default=0.0)
    parser.add_argument("--png-dpi", type=int, default=600)
    parser.add_argument("--panel-cols", type=int, default=3)
    parser.add_argument("--min-footprint-stride", type=int, default=5)
    parser.add_argument("--target-footprints", type=int, default=10)
    parser.add_argument("--focus-width", type=float, default=DEFAULT_FOCUS_WIDTH)
    parser.add_argument("--focus-height", type=float, default=DEFAULT_FOCUS_HEIGHT)
    parser.add_argument("--focus-snap", type=float, default=DEFAULT_FOCUS_SNAP)
    parser.add_argument("--reuse-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(ROOT_DIR) / "log" / "paper_support" / f"dlp_example_gallery_{stamp}"
    )
    ensure_dir(out_dir)

    paper_gallery.apply_reference_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    if args.reuse_dir:
        selected_rows, source_manifest = load_rows_from_existing_output(Path(args.reuse_dir))
        scan_rows = None
    else:
        case_ids = range(int(args.case_start), int(args.case_stop))
        runner = DualCaseRunner(args.forward_ckpt, args.unpark_ckpt)
        selected_rows, scan_rows = select_cases(
            case_ids=case_ids,
            scene_seeds=[int(seed) for seed in args.scene_seeds],
            runner=runner,
            num_cases=int(args.num_cases),
            min_segments=int(args.min_segments),
            max_segments=int(args.max_segments),
            min_steps=int(args.min_steps),
            max_steps=int(args.max_steps),
            min_start_dist=float(args.min_start_dist),
        )
        if not selected_rows:
            raise RuntimeError("No qualified DLP examples were found with the current filters.")

    manifest = {
        "output_dir": str(out_dir),
        "forward_ckpt": args.forward_ckpt,
        "unpark_ckpt": args.unpark_ckpt,
        "scene_seeds": [int(seed) for seed in args.scene_seeds],
        "case_start": int(args.case_start),
        "case_stop": int(args.case_stop),
        "num_cases": int(args.num_cases),
        "min_segments": int(args.min_segments),
        "max_segments": int(args.max_segments),
        "min_steps": int(args.min_steps),
        "max_steps": int(args.max_steps),
        "min_start_dist": float(args.min_start_dist),
        "focus_window": {
            "width": float(args.focus_width),
            "height": float(args.focus_height),
            "snap": float(args.focus_snap),
        },
        "reuse_dir": None if not args.reuse_dir else str(args.reuse_dir),
        "cases": [],
        "panel": None,
    }

    if scan_rows is not None:
        scan_path = out_dir / "scan_rows.json"
        scan_path.write_text(json.dumps(scan_rows, indent=2), encoding="utf-8")

    case_dir = out_dir / "cases"
    ensure_dir(case_dir)
    for row in selected_rows:
        plot_bounds = compute_focus_bounds(
            row["dual"],
            float(args.focus_width),
            float(args.focus_height),
            float(args.focus_snap),
        )
        manifest["cases"].append(
            render_single(
                row,
                plot_bounds,
                case_dir,
                png_dpi=int(args.png_dpi),
                min_stride=int(args.min_footprint_stride),
                target_footprints=int(args.target_footprints),
            )
        )

    manifest["panel"] = render_panel(
        selected_rows,
        out_dir,
        png_dpi=int(args.png_dpi),
        min_stride=int(args.min_footprint_stride),
        target_footprints=int(args.target_footprints),
        panel_cols=int(args.panel_cols),
        focus_width=float(args.focus_width),
        focus_height=float(args.focus_height),
        focus_snap=float(args.focus_snap),
    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out_dir)
    print(manifest_path)


if __name__ == "__main__":
    main()
