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
REPO_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
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

import export_vector_case_plots as vector_plots


DIFFICULTY_ORDER = ("Easy", "Medium", "Hard")

CURATED_CASES = {
    "bay": [
        {
            "name": "bay_easy_normal_seed1818",
            "title": "Normal Bay | Easy",
            "level": "Normal",
            "case_id": 0,
            "slot_type": "Bay",
            "difficulty": "Easy",
            "seed": 1818,
            "action_seed": 2818,
            "note": "Easy bay example with a compact and visually readable maneuver shape.",
            "selection_reason": "Chosen for a clean bay-parking path shape and stable final alignment.",
        },
        {
            "name": "bay_medium_complex_seed1139",
            "title": "Complex Bay | Medium",
            "level": "Complex",
            "case_id": 0,
            "slot_type": "Bay",
            "difficulty": "Medium",
            "seed": 1139,
            "action_seed": 3139,
            "note": "Medium bay case with a smooth-looking connector and uncluttered final insertion.",
            "selection_reason": "Chosen for trajectory readability and a clear bay-slot approach.",
        },
        {
            "name": "bay_hard_complex_seed1634",
            "title": "Complex Bay | Hard",
            "level": "Complex",
            "case_id": 0,
            "slot_type": "Bay",
            "difficulty": "Hard",
            "seed": 1634,
            "action_seed": 3634,
            "note": "Hard bay example whose path remains readable despite a larger maneuver envelope.",
            "selection_reason": "Chosen for a reasonable large-turn bay trajectory and a clear final posture.",
        },
    ],
    "parallel": [
        {
            "name": "parallel_easy_normal_seed1154",
            "title": "Normal Parallel | Easy",
            "level": "Normal",
            "case_id": 1,
            "slot_type": "Parallel",
            "difficulty": "Easy",
            "seed": 1154,
            "action_seed": 102154,
            "note": "Easy parallel case with a long but still visually understandable maneuver sequence.",
            "selection_reason": "Chosen for a reasonable-looking parallel parking path with visible heading evolution.",
        },
        {
            "name": "parallel_medium_complex_seed1820",
            "title": "Complex Parallel | Medium",
            "level": "Complex",
            "case_id": 1,
            "slot_type": "Parallel",
            "difficulty": "Medium",
            "seed": 1820,
            "action_seed": 103820,
            "note": "Medium parallel case selected mainly for a compact and clean path geometry.",
            "selection_reason": "Chosen for figure clarity rather than an aggressive performance gain.",
        },
        {
            "name": "parallel_hard_extrem_seed2383",
            "title": "Extrem Parallel | Hard",
            "level": "Extrem",
            "case_id": 1,
            "slot_type": "Parallel",
            "difficulty": "Hard",
            "seed": 2383,
            "action_seed": 105383,
            "note": "Hard parallel example with a short and visually clean final parked path.",
            "selection_reason": "Chosen for a neat extrem-parallel trajectory that is easy to present in a paper figure.",
        },
    ],
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_result(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_generate_result(case_spec, forward_ckpt: str, unpark_ckpt: str):
    if case_spec.get("trace"):
        trace_path = Path(REPO_DIR) / case_spec["trace"]
        result = load_result(trace_path)
        return result, str(trace_path)

    map_obj = vector_plots.generate_scene_map(case_spec["level"], case_spec["seed"], case_spec["case_id"])
    action_seed = int(case_spec["action_seed"])
    result = vector_plots.run_dual_case(map_obj, forward_ckpt, unpark_ckpt, action_seed)
    source = (
        f"generated:{case_spec['level']}:{case_spec['case_id']}:"
        f"{case_spec['seed']}:{action_seed}"
    )
    return result, source


def apply_reference_style():
    route_color = "#0072B2"
    vector_plots.PALETTE.update(
        {
            "bg": "#FBFBF8",
            "grid": "#E4E1DB",
            "dark": "#3A3A3A",
            "obstacle": "#BDBDBD",
            "start_fill": "none",
            "start_edge": "#00976A",
            "dest_fill": "none",
            "dest_edge": "#D35600",
            "forward_policy": route_color,
            "forward_rs_assist": route_color,
            "rs_connector": route_color,
            "forward_policy_after_connection": route_color,
            "success": route_color,
            "fail": route_color,
        }
    )
    vector_plots.DISPLAY_FOOTPRINT_LINE_COLOR = route_color
    vector_plots.DISPLAY_FOOTPRINT_START_COLOR = "#A06CD5"
    vector_plots.DISPLAY_FOOTPRINT_END_COLOR = "#7B6FD6"
    vector_plots.DISPLAY_FOOTPRINT_EDGE = "#7B6FD6"
    vector_plots.DISPLAY_FOOTPRINT_OUTLINE_ONLY = True
    vector_plots.DISPLAY_SKIP_START_FOOTPRINT = True
    vector_plots.DISPLAY_FOOTPRINT_PATH_LINEWIDTH = 3.00
    vector_plots.DISPLAY_FOOTPRINT_OUTLINE_LINEWIDTH = 1.45
    vector_plots.DISPLAY_FOOTPRINT_OUTLINE_FINAL_LINEWIDTH = 1.70
    vector_plots.UNIFIED_ROUTE_COLOR = route_color
    vector_plots.DISPLAY_CORNER_ROUNDING_PASSES = 6
    vector_plots.DISPLAY_TRAJECTORY_SMOOTH_PASSES = 3
    vector_plots.DISPLAY_DENSE_SAMPLE_SPACING = 0.08
    vector_plots.DISPLAY_RAW_TERMINAL_POINTS = 3

    def reference_tick_step(span):
        if span <= 14:
            return 2.0
        if span <= 36:
            return 4.0
        if span <= 52:
            return 5.0
        return 10.0

    vector_plots.choose_tick_step = reference_tick_step


def case_title(case_spec):
    return f"{case_spec['slot_type']} | {case_spec['difficulty']}"


def snap_down(value: float, step: float):
    return step * math.floor(value / step)


def snap_up(value: float, step: float):
    return step * math.ceil(value / step)


def compute_unified_bounds(results_by_name):
    bounds_list = [result["map"]["bounds"] for result in results_by_name.values()]
    xmin = min(bounds["xmin"] for bounds in bounds_list)
    xmax = max(bounds["xmax"] for bounds in bounds_list)
    ymin = min(bounds["ymin"] for bounds in bounds_list)
    ymax = max(bounds["ymax"] for bounds in bounds_list)
    return {
        "xmin": float(snap_down(xmin, 2.0)),
        "xmax": float(snap_up(xmax, 2.0)),
        "ymin": float(snap_down(ymin, 2.0)),
        "ymax": float(snap_up(ymax, 2.0)),
    }


def clone_result_with_bounds(result, bounds):
    patched = deepcopy(result)
    patched["map"]["bounds"] = {key: float(value) for key, value in bounds.items()}
    return patched


def configure_blank_axes(ax, bounds):
    pad = 1.5
    ax.set_facecolor(vector_plots.PALETTE["bg"])
    ax.grid(True, linestyle="--", linewidth=0.55, color=vector_plots.PALETTE["grid"], alpha=0.45)
    ax.set_xlim(bounds["xmin"] - pad, bounds["xmax"] + pad)
    ax.set_ylim(bounds["ymin"] - pad, bounds["ymax"] + pad)
    ax.set_aspect("equal", adjustable="box")
    x_step = vector_plots.choose_tick_step(bounds["xmax"] - bounds["xmin"])
    y_step = vector_plots.choose_tick_step(bounds["ymax"] - bounds["ymin"])
    x_start = x_step * np.floor(bounds["xmin"] / x_step)
    y_start = y_step * np.floor(bounds["ymin"] / y_step)
    ax.set_xticks(np.arange(x_start, bounds["xmax"] + x_step, x_step))
    ax.set_yticks(np.arange(y_start, bounds["ymax"] + y_step, y_step))
    ax.tick_params(length=2.6, width=0.8, labelsize=14.0, colors=vector_plots.PALETTE["dark"])
    for spine_name, spine in ax.spines.items():
        if spine_name in ("left", "bottom"):
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color(vector_plots.PALETTE["grid"])
        else:
            spine.set_visible(False)


def choose_stride(num_states: int, min_stride: int, target_footprints: int):
    if num_states <= 1:
        return 1
    adaptive = int(math.ceil((num_states - 1) / max(target_footprints, 1)))
    return max(int(min_stride), adaptive)


def concise_status_text(result):
    status_text = "ARRIVED" if result["final_success"] else result["status"]
    conn_text = ""
    if result.get("connection_used") is not None:
        conn_text = f", conn={int(bool(result['connection_used']))}"
        if result.get("connection_index") is not None:
            conn_text += f", idx={result['connection_index']}"
    return f"{status_text}, steps={result['step_num']}{conn_text}"


def render_single_case(case_spec, result, plot_bounds, out_dir: Path, png_dpi: int, min_stride: int, target_footprints: int):
    result_for_plot = clone_result_with_bounds(result, plot_bounds)
    stride = choose_stride(len(result["trajectory_states"]), min_stride=min_stride, target_footprints=target_footprints)

    fig, ax = plt.subplots(figsize=(6.8, 5.7))
    vector_plots.draw_case(
        ax,
        case_title(case_spec),
        result_for_plot,
        show_legend=False,
        show_axis_labels=True,
        trajectory_style="footprints",
        footprint_stride=stride,
    )
    ax.set_title(case_title(case_spec), fontsize=16.5, color=vector_plots.PALETTE["dark"])
    ax.set_xlabel("x (m)", fontsize=15.0, color=vector_plots.PALETTE["dark"])
    ax.set_ylabel("y (m)", fontsize=15.0, color=vector_plots.PALETTE["dark"])
    ax.tick_params(labelsize=18.0)

    base = out_dir / case_spec["name"]
    fig.savefig(base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    return {
        "name": case_spec["name"],
        "title": case_spec["title"],
        "stride": stride,
        "status": result["status"],
        "final_success": bool(result["final_success"]),
        "planning_success": bool(result.get("planning_success", result["final_success"])),
        "connection_used": bool(result.get("connection_used", False)),
        "connection_index": result.get("connection_index"),
        "step_num": int(result["step_num"]),
        "level": case_spec["level"],
        "slot_type": case_spec["slot_type"],
        "difficulty": case_spec["difficulty"],
        "seed": case_spec["seed"],
        "note": case_spec["note"],
    }


def render_panel(group_name: str, case_specs, results_by_name, plot_bounds, out_dir: Path, png_dpi: int, min_stride: int, target_footprints: int):
    fig, axes = plt.subplots(nrows=1, ncols=len(DIFFICULTY_ORDER), figsize=(18.8, 6.4))
    if len(DIFFICULTY_ORDER) == 1:
        axes = [axes]

    difficulty_to_case = {case_spec["difficulty"]: case_spec for case_spec in case_specs}
    manifest_rows = []
    for idx, (ax, difficulty) in enumerate(zip(axes, DIFFICULTY_ORDER)):
        case_spec = difficulty_to_case.get(difficulty)
        if case_spec is None:
            configure_blank_axes(ax, plot_bounds)
            ax.set_title(f"{group_name.title()} | {difficulty}", fontsize=16.0, color=vector_plots.PALETTE["dark"])
            ax.set_xlabel("x (m)", fontsize=14.5, color=vector_plots.PALETTE["dark"])
            ax.set_ylabel("y (m)" if idx == 0 else "", fontsize=14.5, color=vector_plots.PALETTE["dark"])
            ax.tick_params(labelsize=17.0)
            manifest_rows.append({"name": None, "difficulty": difficulty, "stride": None})
            continue

        result = results_by_name[case_spec["name"]]
        result_for_plot = clone_result_with_bounds(result, plot_bounds)
        stride = choose_stride(len(result["trajectory_states"]), min_stride=min_stride, target_footprints=target_footprints)
        vector_plots.draw_case(
            ax,
            case_title(case_spec),
            result_for_plot,
            show_legend=False,
            show_axis_labels=True,
            trajectory_style="footprints",
            footprint_stride=stride,
        )
        ax.set_title(case_title(case_spec), fontsize=16.5, color=vector_plots.PALETTE["dark"])
        ax.set_xlabel("x (m)", fontsize=14.5, color=vector_plots.PALETTE["dark"])
        if idx == 0:
            ax.set_ylabel("y (m)", fontsize=14.5, color=vector_plots.PALETTE["dark"])
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=17.0)
        manifest_rows.append({"name": case_spec["name"], "difficulty": difficulty, "stride": stride})

    fig.tight_layout()
    base = out_dir / f"{group_name}_panel"
    fig.savefig(base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description="Export paper-ready dual-model planning figures for bay/parallel parking.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--png-dpi", type=int, default=600)
    parser.add_argument("--min-footprint-stride", type=int, default=5)
    parser.add_argument("--target-footprints", type=int, default=10)
    parser.add_argument("--forward-ckpt", type=str, default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--unpark-ckpt", type=str, default="src/log/exp/unpark_ppo_20260323_120610/PPO_unpark_best.pt")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = Path(args.output_dir) if args.output_dir else Path(ROOT_DIR) / "log" / "paper_support" / f"dual_parking_paper_gallery_{stamp}"
    ensure_dir(out_dir)
    apply_reference_style()

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

    manifest = {
        "output_dir": str(out_dir),
        "png_dpi": args.png_dpi,
        "min_footprint_stride": args.min_footprint_stride,
        "target_footprints": args.target_footprints,
        "forward_ckpt": args.forward_ckpt,
        "unpark_ckpt": args.unpark_ckpt,
        "groups": {},
        "notes": [
            "Figures are rendered from curated dual-model cases using the footprint style and a paper-oriented reference palette.",
            "The selected cases prioritize trajectories that look geometrically reasonable and read clearly in paper figures.",
            "Hard bay uses a challenging Complex bay case because Extrem bay is not defined in the released benchmark maps.",
        ],
    }

    results_by_name = {}
    result_sources = {}
    for case_specs in CURATED_CASES.values():
        for case_spec in case_specs:
            result, result_source = load_or_generate_result(case_spec, args.forward_ckpt, args.unpark_ckpt)
            results_by_name[case_spec["name"]] = result
            result_sources[case_spec["name"]] = result_source
    unified_bounds = compute_unified_bounds(results_by_name)
    manifest["unified_plot_bounds"] = unified_bounds
    manifest["result_sources"] = result_sources

    for group_name, case_specs in CURATED_CASES.items():
        group_dir = out_dir / group_name
        ensure_dir(group_dir)
        rows = []
        for case_spec in case_specs:
            rows.append(
                render_single_case(
                    case_spec,
                    results_by_name[case_spec["name"]],
                    unified_bounds,
                    group_dir,
                    png_dpi=args.png_dpi,
                    min_stride=args.min_footprint_stride,
                    target_footprints=args.target_footprints,
                )
            )
            rows[-1]["source"] = result_sources[case_spec["name"]]
            rows[-1]["selection_reason"] = case_spec.get("selection_reason")
        panel_rows = render_panel(
            group_name,
            case_specs,
            results_by_name,
            unified_bounds,
            out_dir,
            png_dpi=args.png_dpi,
            min_stride=args.min_footprint_stride,
            target_footprints=args.target_footprints,
        )
        manifest["groups"][group_name] = {"cases": rows, "panel": panel_rows}

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out_dir)
    print(manifest_path)


if __name__ == "__main__":
    main()
