import argparse
import json
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

import export_dual_parking_paper_gallery as paper_gallery
from analysis.export_vector_case_plots import (
    ensure_dir,
    generate_scene_map,
    map_to_dict,
    run_dual_case,
)


SCAN_SPECS = [
    {"level": "Normal", "case_id": 0, "name": "Normal Bay", "tag": "normal_bay", "folder": "normal_bay"},
    {"level": "Complex", "case_id": 0, "name": "Complex Bay", "tag": "complex_bay", "folder": "complex_bay"},
    {"level": "Normal", "case_id": 1, "name": "Normal Parallel", "tag": "normal_parallel", "folder": "normal_parallel"},
    {"level": "Complex", "case_id": 1, "name": "Complex Parallel", "tag": "complex_parallel", "folder": "complex_parallel"},
    {"level": "Extrem", "case_id": 1, "name": "Extrem Parallel", "tag": "extrem_parallel", "folder": "extrem_parallel"},
]

DEFAULT_CANDIDATE_ROOT = (
    Path(ROOT_DIR)
    / "log"
    / "analysis"
    / "paper_display_smooth_gallery_fixed_20260330_151611"
    / "random_bay_parallel_suite_hd_more_20260330_165616"
)


def sign_of_speed(action):
    speed = float(action[1])
    if speed > 1e-6:
        return 1
    if speed < -1e-6:
        return -1
    return 0


def _extract_motion_runs(segments):
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
    runs = _extract_motion_runs(segments)
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


def default_action_seed(case_id, seed):
    return 41000 + case_id * 1000 + seed


def format_elapsed_ms(elapsed_ms):
    if elapsed_ms is None:
        return "n/a"
    elapsed_ms = float(elapsed_ms)
    if elapsed_ms >= 1000.0:
        return f"{elapsed_ms / 1000.0:.2f}s"
    return f"{elapsed_ms:.0f}ms"


def load_candidate_seeds(spec, candidate_root: Path):
    index_path = candidate_root / spec["folder"] / "index.json"
    if not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return [int(item["seed"]) for item in payload.get("cases", [])]


def evaluate_seed(spec, seed, forward_ckpt, unpark_ckpt):
    map_obj = generate_scene_map(spec["level"], seed, spec["case_id"])
    action_seed = default_action_seed(spec["case_id"], seed)
    dual = run_dual_case(map_obj, forward_ckpt, unpark_ckpt, action_seed)
    motion_segments, raw_motion_segments = count_motion_segments(dual["segments"])
    return {
        "level": spec["level"],
        "case_id": spec["case_id"],
        "name": spec["name"],
        "tag": spec["tag"],
        "seed": seed,
        "action_seed": action_seed,
        "dual": dual,
        "map": map_to_dict(map_obj),
        "motion_segments": motion_segments,
        "raw_motion_segments": raw_motion_segments,
        "connection_used": bool(dual.get("connection_used", False)),
        "connection_index": dual.get("connection_index"),
    }


def candidate_score(row):
    # Prefer simpler successful plans with fewer steps, while allowing both
    # two-segment and three-segment solutions.
    return (
        1000
        - 120 * abs(row["motion_segments"] - 2)
        - float(row["dual"]["step_num"])
        - (0.5 if row["connection_used"] else 0.0)
    )


def qualify(row, min_segments, max_segments, max_steps):
    if not row["dual"]["final_success"]:
        return False
    if row["motion_segments"] < min_segments or row["motion_segments"] > max_segments:
        return False
    if row["dual"]["step_num"] > max_steps:
        return False
    return True


def select_cases_for_spec(
    spec,
    forward_ckpt,
    unpark_ckpt,
    per_spec,
    min_segments,
    max_segments,
    max_steps,
    max_seed,
    candidate_root: Path,
):
    seen = set()
    rows = []

    def maybe_eval(seed):
        if seed in seen:
            return
        seen.add(seed)
        row = evaluate_seed(spec, seed, forward_ckpt, unpark_ckpt)
        if qualify(row, min_segments=min_segments, max_segments=max_segments, max_steps=max_steps):
            row["score"] = candidate_score(row)
            rows.append(row)

    for seed in load_candidate_seeds(spec, candidate_root):
        maybe_eval(seed)

    if len(rows) < per_spec:
        for seed in range(1, max_seed + 1):
            maybe_eval(seed)
            if len(rows) >= per_spec:
                break

    rows.sort(
        key=lambda row: (
            abs(row["motion_segments"] - 2),
            row["dual"]["step_num"],
            -row["score"],
            row["seed"],
        )
    )
    return rows


def render_panel(spec, rows, out_dir: Path, png_dpi: int, min_stride: int, target_footprints: int):
    results_by_name = {}
    case_specs = []
    for idx, row in enumerate(rows, start=1):
        name = f"{spec['tag']}_seed{row['seed']}"
        case_spec = {
            "name": name,
            "title": f"{spec['name']} #{idx}",
            "slot_type": spec["name"],
            "difficulty": f"No.{idx}",
        }
        case_specs.append(case_spec)
        results_by_name[name] = row["dual"]

    bounds = paper_gallery.compute_unified_bounds(results_by_name)
    fig, axes = plt.subplots(nrows=1, ncols=len(rows), figsize=(5.8 * len(rows), 6.4))
    if len(rows) == 1:
        axes = [axes]

    manifest_rows = []
    for idx, (ax, row, case_spec) in enumerate(zip(axes, rows, case_specs)):
        result_for_plot = row["dual"]
        stride = paper_gallery.choose_stride(
            len(row["dual"]["trajectory_states"]),
            min_stride=min_stride,
            target_footprints=target_footprints,
        )
        paper_gallery.vector_plots.draw_case(
            ax,
            spec["name"],
            result_for_plot,
            show_legend=False,
            show_axis_labels=True,
            trajectory_style="footprints",
            footprint_stride=stride,
        )
        ax.set_title("")
        ax.set_xlabel("x (m)", fontsize=14.0, color=paper_gallery.vector_plots.PALETTE["dark"])
        ax.set_ylabel("y (m)" if idx == 0 else "", fontsize=14.0, color=paper_gallery.vector_plots.PALETTE["dark"])
        ax.tick_params(labelsize=19.0)
        manifest_rows.append(
            {
                "name": case_spec["name"],
                "seed": row["seed"],
                "motion_segments": row["motion_segments"],
                "raw_motion_segments": row["raw_motion_segments"],
                "step_num": int(row["dual"]["step_num"]),
                "inference_time_ms": float(row["dual"].get("inference_time_ms", 0.0)),
                "avg_step_time_ms": float(row["dual"].get("avg_step_time_ms", 0.0)),
                "stride": stride,
            }
        )

    fig.tight_layout()
    base = out_dir / f"{spec['tag']}_panel"
    fig.savefig(base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return manifest_rows, bounds


def export_cases(spec, rows, out_dir: Path, png_dpi: int, min_stride: int, target_footprints: int):
    case_dir = out_dir / spec["tag"]
    ensure_dir(case_dir)
    results_by_name = {f"{spec['tag']}_seed{row['seed']}": row["dual"] for row in rows}
    bounds = paper_gallery.compute_unified_bounds(results_by_name)

    exported = []
    for idx, row in enumerate(rows, start=1):
        name = f"{spec['tag']}_seed{row['seed']}"
        result_for_plot = paper_gallery.clone_result_with_bounds(row["dual"], bounds)
        stride = paper_gallery.choose_stride(
            len(row["dual"]["trajectory_states"]),
            min_stride=min_stride,
            target_footprints=target_footprints,
        )
        fig, ax = plt.subplots(figsize=(6.8, 5.7))
        paper_gallery.vector_plots.draw_case(
            ax,
            spec["name"],
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
        base = case_dir / name
        fig.savefig(base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

        summary = {
            "scene_name": spec["name"],
            "level": row["level"],
            "case_id": row["case_id"],
            "seed": row["seed"],
            "action_seed": row["action_seed"],
            "motion_segments": row["motion_segments"],
            "raw_motion_segments": row["raw_motion_segments"],
            "step_num": int(row["dual"]["step_num"]),
            "inference_time_ms": float(row["dual"].get("inference_time_ms", 0.0)),
            "prepare_time_ms": float(row["dual"].get("prepare_time_ms", 0.0)),
            "rollout_time_ms": float(row["dual"].get("rollout_time_ms", 0.0)),
            "avg_step_time_ms": float(row["dual"].get("avg_step_time_ms", 0.0)),
            "connection_used": row["connection_used"],
            "connection_index": row["connection_index"],
            "score": row["score"],
        }
        (case_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (case_dir / f"{name}_trace.json").write_text(json.dumps(row["dual"], indent=2), encoding="utf-8")
        exported.append(summary)

    panel_rows, panel_bounds = render_panel(
        spec,
        rows,
        out_dir,
        png_dpi=png_dpi,
        min_stride=min_stride,
        target_footprints=target_footprints,
    )
    return {
        "cases": exported,
        "panel": panel_rows,
        "plot_bounds": panel_bounds,
    }


def main():
    parser = argparse.ArgumentParser(description="Select good dual-model planning cases with 2-3 motion segments.")
    parser.add_argument("--forward_ckpt", type=str, default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--unpark_ckpt", type=str, default="src/log/exp/unpark_ppo_20260323_120610/PPO_unpark_best.pt")
    parser.add_argument("--per_spec", type=int, default=5)
    parser.add_argument("--min_segments", type=int, default=2)
    parser.add_argument("--max_segments", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--max_seed", type=int, default=3000)
    parser.add_argument("--candidate_root", type=str, default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--png_dpi", type=int, default=600)
    parser.add_argument("--min_footprint_stride", type=int, default=5)
    parser.add_argument("--target_footprints", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(ROOT_DIR) / "log" / "paper_support" / f"dual_segment_gallery_{stamp}"
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

    candidate_root = Path(args.candidate_root)
    manifest = {
        "output_dir": str(out_dir),
        "forward_ckpt": args.forward_ckpt,
        "unpark_ckpt": args.unpark_ckpt,
        "per_spec": args.per_spec,
        "min_segments": args.min_segments,
        "max_segments": args.max_segments,
        "max_steps": args.max_steps,
        "max_seed": args.max_seed,
        "candidate_root": str(candidate_root),
        "scenes": {},
    }

    for spec_idx, spec in enumerate(SCAN_SPECS, start=1):
        print(f"[{spec_idx}/{len(SCAN_SPECS)}] scan {spec['name']}")
        rows = select_cases_for_spec(
            spec,
            forward_ckpt=args.forward_ckpt,
            unpark_ckpt=args.unpark_ckpt,
            per_spec=args.per_spec,
            min_segments=args.min_segments,
            max_segments=args.max_segments,
            max_steps=args.max_steps,
            max_seed=args.max_seed,
            candidate_root=candidate_root,
        )
        if len(rows) < args.per_spec:
            print(f"warning: {spec['name']} only found {len(rows)} qualified cases")
        manifest["scenes"][spec["tag"]] = export_cases(
            spec,
            rows[: args.per_spec],
            out_dir,
            png_dpi=args.png_dpi,
            min_stride=args.min_footprint_stride,
            target_footprints=args.target_footprints,
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out_dir)
    print(manifest_path)


if __name__ == "__main__":
    main()
