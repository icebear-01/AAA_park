import argparse
import json
import math
import os
import sys
import time
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
from matplotlib.patches import Polygon as PolygonPatch

from export_failure_efficiency_examples import (
    PALETTE,
    apply_bounds,
    build_sac_agent,
    create_pure_rl_env,
    generate_scene_map,
    run_pure_rl_case,
    save_figure,
    state_box_coords,
)


SCENE_SPECS = [
    {
        "key": "complex_bay",
        "label": "Complex Bay",
        "level": "Complex",
        "map_case_id": 0,
        "orientation": "vertical",
        "candidate_seeds": [36, 57, 22, 55, 10, 1634, 2601, 2567, 2974],
    },
    {
        "key": "complex_parallel",
        "label": "Complex Parallel",
        "level": "Complex",
        "map_case_id": 1,
        "orientation": "horizontal",
        "candidate_seeds": [99, 111, 9, 13, 2755, 94, 26, 68, 20],
    },
    {
        "key": "extrem_parallel",
        "label": "Extrem Parallel",
        "level": "Extrem",
        "map_case_id": 1,
        "orientation": "horizontal_extreme",
        "candidate_seeds": [7, 52, 6263, 41, 1872, 3, 17, 29, 63],
    },
]

FAILURE_COLOR = "#D55E00"
FAILURE_FILL = (0.86, 0.26, 0.11, 0.28)
FAILURE_FINAL_FILL = (0.86, 0.26, 0.11, 0.48)


def angle_wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def final_error(result):
    final_state = result["trajectory_states"][-1]
    dest = result["map"]["dest"]
    dist = math.hypot(final_state["x"] - dest["x"], final_state["y"] - dest["y"])
    heading = abs(angle_wrap(final_state["heading"] - dest["heading"]))
    heading = min(heading, math.pi - heading) if heading > math.pi / 2 else heading
    return dist, heading


def max_error(result):
    dest = result["map"]["dest"]
    distances = [
        math.hypot(state["x"] - dest["x"], state["y"] - dest["y"])
        for state in result["trajectory_states"]
    ]
    return max(distances) if distances else 0.0


def failure_score(result):
    dist, heading = final_error(result)
    status_bonus = {
        "COLLIDED": 12.0,
        "OUTBOUND": 10.0,
        "OUTTIME": 4.0,
    }.get(result["status"], 0.0)
    return status_bonus + 2.2 * dist + 2.5 * heading + 0.012 * result["step_num"] + 0.18 * max_error(result)


def is_clear_failure(result, min_final_dist, min_heading_deg, allow_collision):
    if result["final_success"] or result["status"] == "ARRIVED":
        return False
    dist, heading = final_error(result)
    if allow_collision and result["status"] in {"COLLIDED", "OUTBOUND"}:
        return True
    if result["status"] == "OUTTIME" and dist >= min_final_dist:
        return True
    if result["status"] == "OUTTIME" and math.degrees(heading) >= min_heading_deg and dist >= 1.0:
        return True
    return False


def passes_status_filter(result, status_filter):
    if not status_filter:
        return True
    allowed = {item.strip().upper() for item in status_filter.split(",") if item.strip()}
    return result["status"].upper() in allowed


def seed_sequence(spec, max_seed):
    seen = set()
    for seed in spec["candidate_seeds"]:
        if seed not in seen:
            seen.add(seed)
            yield seed
    for seed in range(max_seed):
        if seed not in seen:
            seen.add(seed)
            yield seed


def draw_clear_failure(ax, result, title):
    states = result["trajectory_states"]
    points = np.asarray([[state["x"], state["y"]] for state in states], dtype=np.float64)
    apply_bounds(ax, result["map"], points.tolist())
    if len(points) >= 2:
        ax.plot(points[:, 0], points[:, 1], color="white", linewidth=5.2, alpha=0.75, zorder=7)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=FAILURE_COLOR,
            linewidth=3.2,
            alpha=0.96,
            zorder=8,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    footprint_indices = [0]
    if len(states) > 1:
        footprint_indices.append(len(states) - 1)
    for order_idx, idx in enumerate(footprint_indices):
        state = states[idx]
        is_final = idx == footprint_indices[-1]
        ax.add_patch(
            PolygonPatch(
                state_box_coords(state),
                closed=True,
                facecolor=FAILURE_FILL if not is_final else FAILURE_FINAL_FILL,
                edgecolor=FAILURE_COLOR,
                linewidth=1.55 if not is_final else 2.0,
                zorder=10 + order_idx / max(len(footprint_indices), 1),
            )
        )

    start = states[0]
    final = states[-1]
    ax.scatter([start["x"]], [start["y"]], s=52, color=FAILURE_COLOR, edgecolor="white", linewidth=0.8, zorder=16)
    ax.scatter([final["x"]], [final["y"]], s=70, color=FAILURE_COLOR, edgecolor="white", linewidth=0.8, zorder=17)
    dist, heading = final_error(result)
    ax.set_title(title, fontsize=18, color=PALETTE["dark"], pad=10, fontweight="semibold")
    ax.text(
        0.985,
        0.985,
        f"{result['status']}\nsteps {result['step_num']}\nfinal error {dist:.1f} m",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=12.0,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#E5E7EB", "boxstyle": "round,pad=0.22", "alpha": 0.90},
        zorder=30,
    )


def render_case(item, output_dir, dpi):
    case = item["case"]
    result = item["result"]
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    draw_clear_failure(ax, result, f"{case['label']} seed {case['scene_seed']} | single model failure")
    save_figure(fig, output_dir / "figures" / f"{case['case_key']}_single_model_failure", dpi)


def render_overview(items, output_dir, dpi):
    rows = int(math.ceil(len(items) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13.6, 5.1 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, item in zip(axes, items):
        case = item["case"]
        draw_clear_failure(ax, item["result"], f"{case['label']} seed {case['scene_seed']}")
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle("Single Model Clear Failure Cases", fontsize=20, color=PALETTE["dark"], fontweight="semibold")
    save_figure(fig, output_dir / "overview_single_model_clear_failure_cases", dpi)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def scan_cases(args):
    env = create_pure_rl_env()
    agent = build_sac_agent(args.forward_ckpt, env)
    env.close()

    selected = []
    specs = SCENE_SPECS
    if args.scene_filter:
        requested = {item.strip() for item in args.scene_filter.split(",") if item.strip()}
        specs = [spec for spec in SCENE_SPECS if spec["key"] in requested]
    for spec in specs:
        candidates = []
        attempts = 0
        print(f"[scan] {spec['label']}", flush=True)
        for scene_seed in seed_sequence(spec, args.max_seed):
            attempts += 1
            map_obj = generate_scene_map(spec["level"], scene_seed, spec["map_case_id"])
            action_seed = int(args.action_seed_base + scene_seed + 10000 * spec["map_case_id"])
            result = run_pure_rl_case(
                map_obj,
                agent,
                action_seed,
                args.rl_action_mode,
                action_mask_input_mode=args.action_mask_input,
                terminate_on_collision=args.terminate_on_collision,
            )
            if passes_status_filter(result, args.status_filter) and is_clear_failure(
                result,
                args.min_final_dist,
                args.min_heading_deg,
                args.allow_collision,
            ):
                dist, heading = final_error(result)
                case = {
                    "case_key": f"{spec['key']}_seed{scene_seed}",
                    "label": spec["label"],
                    "orientation": spec["orientation"],
                    "level": spec["level"],
                    "map_case_id": spec["map_case_id"],
                    "scene_seed": int(scene_seed),
                    "action_seed": int(action_seed),
                    "final_distance": float(dist),
                    "final_heading_error_deg": float(math.degrees(heading)),
                    "score": float(failure_score(result)),
                }
                candidates.append({"case": case, "result": result})
                print(
                    f"[candidate] {case['case_key']} status={result['status']} "
                    f"steps={result['step_num']} dist={dist:.2f} heading={math.degrees(heading):.1f}"
                    ,
                    flush=True,
                )
            if len(candidates) >= args.per_spec or attempts >= args.max_seed + len(spec["candidate_seeds"]):
                break
        candidates.sort(key=lambda item: item["case"]["score"], reverse=True)
        picked = candidates[: args.per_spec]
        selected.extend(picked)
        print(f"[scan] {spec['label']} picked={len(picked)} attempts={attempts}", flush=True)
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Export visually clear single-model failure cases.")
    parser.add_argument("--forward-ckpt", default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--output-dir", default="src/log/paper_support/single_model_clear_failure_cases_hidpi")
    parser.add_argument("--per-spec", type=int, default=2)
    parser.add_argument("--max-seed", type=int, default=300)
    parser.add_argument("--action-seed-base", type=int, default=91000)
    parser.add_argument("--rl-action-mode", choices=["raw", "choose"], default="raw")
    parser.add_argument("--min-final-dist", type=float, default=2.5)
    parser.add_argument("--min-heading-deg", type=float, default=45.0)
    parser.add_argument("--allow-collision", action="store_true", default=True)
    parser.add_argument("--png-dpi", type=int, default=600)
    parser.add_argument("--scene-filter", default="")
    parser.add_argument("--action-mask-input", choices=["actual", "ones", "zeros"], default="actual")
    parser.add_argument("--terminate-on-collision", action="store_true")
    parser.add_argument("--status-filter", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = scan_cases(args)
    for item in items:
        render_case(item, output_dir, args.png_dpi)
        write_json(output_dir / "case_json" / f"{item['case']['case_key']}_single_model_failure.json", item)
    render_overview(items, output_dir, args.png_dpi)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "forward_ckpt": str(Path(args.forward_ckpt).resolve()),
        "output_dir": str(output_dir),
        "definition": "Single forward SAC model with RS assist disabled. Cases are selected only when the rollout fails visibly: collision/outbound, or timeout with large terminal error.",
        "args": vars(args),
        "cases": [
            {
                **item["case"],
                "status": item["result"]["status"],
                "steps": item["result"]["step_num"],
                "final_success": item["result"]["final_success"],
                "rs_assist_used": item["result"]["rs_assist_used"],
                "action_mode": item["result"]["action_mode"],
                "action_mask_input_mode": item["result"].get("action_mask_input_mode", "actual"),
                "terminate_on_collision": item["result"].get("terminate_on_collision", False),
            }
            for item in items
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"[done] output_dir={output_dir}")
    print(f"[done] cases={len(items)}")


if __name__ == "__main__":
    main()
