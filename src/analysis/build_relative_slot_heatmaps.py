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

from analysis.build_dual_framework_paper_package import (
    DualRunner,
    PALETTE,
    SingleRunner,
    ensure_dir,
    save_figure,
    scene_title,
    setup_style,
    style_axis,
)
from analysis.export_vector_case_plots import generate_scene_map


def slot_frame_transform(point_xy, slot_center, slot_heading):
    dx = point_xy[0] - slot_center[0]
    dy = point_xy[1] - slot_center[1]
    cos_h = math.cos(-slot_heading)
    sin_h = math.sin(-slot_heading)
    return np.array(
        [
            cos_h * dx - sin_h * dy,
            sin_h * dx + cos_h * dy,
        ],
        dtype=np.float64,
    )


def slot_frame_box(dest_box, slot_center, slot_heading):
    return np.array(
        [slot_frame_transform(point, slot_center, slot_heading) for point in np.asarray(dest_box, dtype=np.float64)],
        dtype=np.float64,
    )


def midpoint_samples(case_result):
    map_info = case_result["map"]
    dest_box = np.asarray(map_info["dest_box"], dtype=np.float64)
    slot_center = dest_box.mean(axis=0)
    slot_heading = float(map_info["dest"]["heading"])
    samples = []
    for segment in case_result["segments"]:
        midpoint = np.array(
            [
                0.5 * (segment["x0"] + segment["x1"]),
                0.5 * (segment["y0"] + segment["y1"]),
            ],
            dtype=np.float64,
        )
        samples.append(slot_frame_transform(midpoint, slot_center, slot_heading))
    return np.array(samples, dtype=np.float64), slot_frame_box(dest_box, slot_center, slot_heading)


def scan_scene_records(level, case_id, forward_ckpt, unpark_ckpt, max_seed):
    single_runner = SingleRunner(forward_ckpt)
    dual_runner = DualRunner(forward_ckpt, unpark_ckpt)
    rows = []
    try:
        for seed in range(1, max_seed + 1):
            map_obj = generate_scene_map(level, seed, case_id)
            action_seed = 31000 + case_id * 1000 + seed
            single = single_runner.run(map_obj, action_seed, capture_trace=False)
            dual = dual_runner.run(map_obj, action_seed, capture_trace=False)
            step_gain = single["step_num"] - dual["step_num"]
            rows.append(
                {
                    "seed": seed,
                    "case_id": case_id,
                    "single_success": single["final_success"],
                    "single_status": single["status"],
                    "single_steps": single["step_num"],
                    "dual_success": dual["final_success"],
                    "dual_status": dual["status"],
                    "dual_steps": dual["step_num"],
                    "step_gain": step_gain,
                }
            )
    finally:
        single_runner.close()
        dual_runner.close()
    return rows


def select_case_rows(rows, selection_mode, min_single_steps, min_step_gain, max_cases_per_type):
    selected = []
    for row in rows:
        rescued = (not row["single_success"]) and row["dual_success"]
        accelerated = row["single_success"] and row["dual_success"] and row["step_gain"] >= min_step_gain
        single_hard = (not row["single_success"]) or row["single_steps"] >= min_single_steps
        dual_better = row["dual_success"] and (rescued or row["step_gain"] >= min_step_gain)

        row = {
            **row,
            "rescued": rescued,
            "accelerated": accelerated,
            "single_hard": single_hard,
            "dual_better": dual_better,
        }

        keep = False
        if selection_mode == "improved":
            keep = rescued or accelerated
        elif selection_mode == "single_hard":
            keep = single_hard
        elif selection_mode == "single_hard_dual_better":
            keep = single_hard and dual_better
        else:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")

        if keep:
            selected.append(row)

    if selection_mode == "improved":
        selected.sort(key=lambda row: (not row["rescued"], -row["step_gain"], row["seed"]))
    elif selection_mode == "single_hard":
        selected.sort(key=lambda row: (row["single_success"], -row["single_steps"], -row["step_gain"], row["seed"]))
    else:
        selected.sort(key=lambda row: (not row["rescued"], row["single_success"], -row["step_gain"], -row["single_steps"], row["seed"]))

    if max_cases_per_type > 0:
        selected = selected[:max_cases_per_type]
    return selected


def capture_cases(level, case_id, seeds, forward_ckpt, unpark_ckpt):
    single_runner = SingleRunner(forward_ckpt)
    dual_runner = DualRunner(forward_ckpt, unpark_ckpt)
    cases = []
    try:
        for seed in seeds:
            map_obj = generate_scene_map(level, seed, case_id)
            action_seed = 33000 + case_id * 1000 + seed
            single = single_runner.run(map_obj, action_seed, capture_trace=True)
            dual = dual_runner.run(map_obj, action_seed, capture_trace=True)
            cases.append(
                {
                    "seed": seed,
                    "case_id": case_id,
                    "single": single,
                    "dual": dual,
                }
            )
    finally:
        single_runner.close()
        dual_runner.close()
    return cases


def collect_heatmap_samples(cases, method_key):
    all_points = []
    box = None
    for case in cases:
        points, slot_box = midpoint_samples(case[method_key])
        if len(points) > 0:
            all_points.append(points)
        if box is None:
            box = slot_box
    if not all_points:
        return np.zeros((0, 2), dtype=np.float64), box
    return np.concatenate(all_points, axis=0), box


def compute_extent(point_sets, slot_box):
    arrays = [pts for pts in point_sets if pts is not None and len(pts) > 0]
    if slot_box is not None:
        arrays.append(slot_box)
    merged = np.concatenate(arrays, axis=0)
    xmin, ymin = merged.min(axis=0)
    xmax, ymax = merged.max(axis=0)
    xpad = max(1.0, 0.12 * (xmax - xmin + 1e-6))
    ypad = max(1.0, 0.12 * (ymax - ymin + 1e-6))
    return [xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad]


def count_hist(points, bins, extent):
    if points is None or len(points) == 0:
        return np.zeros((bins, bins), dtype=np.float64)
    hist, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
    )
    return hist


def draw_heatmap(ax, hist, extent, slot_box, title, vmax):
    style_axis(ax, y_grid=False)
    ax.set_facecolor("#FBFBF8")
    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=extent,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax if vmax > 0 else None,
        interpolation="bicubic",
        aspect="equal",
    )
    if slot_box is not None:
        ax.add_patch(
            PolygonPatch(
                slot_box,
                closed=True,
                facecolor="#DDEBD3",
                edgecolor="#648A46",
                linewidth=2.0,
                alpha=0.95,
                zorder=5,
            )
        )
    ax.scatter([0.0], [0.0], s=36, color="#223044", zorder=6)
    ax.set_title(title, fontsize=12.5, color=PALETTE["dark"])
    ax.set_xlabel("Relative x to target slot (m)")
    ax.set_ylabel("Relative y to target slot (m)")
    return im


def plot_slot_relative_heatmaps(level, bay_cases, parallel_cases, out_dir, png_dpi):
    setup_style()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.8, 10.8))

    row_specs = [
        ("Bay", bay_cases),
        ("Parallel", parallel_cases),
    ]
    manifest = {}

    for row_idx, (ptype, cases) in enumerate(row_specs):
        single_points, slot_box = collect_heatmap_samples(cases, "single")
        dual_points, _ = collect_heatmap_samples(cases, "dual")
        extent = compute_extent([single_points, dual_points], slot_box)
        single_hist = count_hist(single_points, bins=56, extent=extent)
        dual_hist = count_hist(dual_points, bins=56, extent=extent)
        vmax = max(float(single_hist.max()), float(dual_hist.max()))

        im_left = draw_heatmap(
            axes[row_idx, 0],
            single_hist,
            extent,
            slot_box,
            f"{ptype} slot | HOPE",
            vmax,
        )
        im_right = draw_heatmap(
            axes[row_idx, 1],
            dual_hist,
            extent,
            slot_box,
            f"{ptype} slot | Dual model",
            vmax,
        )

        cbar = fig.colorbar(im_right, ax=axes[row_idx, :], fraction=0.025, pad=0.02)
        cbar.set_label("Visit count")
        manifest[ptype] = {
            "num_cases": len(cases),
            "single_points": int(len(single_points)),
            "dual_points": int(len(dual_points)),
            "seed_list": [case["seed"] for case in cases],
        }

    fig.suptitle(
        f"{scene_title(level)} hard-case action density in the target-slot frame",
        fontsize=15,
        y=0.995,
        color=PALETTE["dark"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save_figure(fig, out_dir / "fig07_relative_slot_heatmaps", png_dpi)
    (out_dir / "fig07_relative_slot_heatmaps_manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward_ckpt", type=str, default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--unpark_ckpt", type=str, default="src/log/exp/unpark_ppo_20260323_120610/PPO_unpark_best.pt")
    parser.add_argument("--level", type=str, default="Complex")
    parser.add_argument("--max_seed", type=int, default=100)
    parser.add_argument("--max_cases_per_type", type=int, default=40)
    parser.add_argument("--selection_mode", type=str, default="single_hard", choices=["improved", "single_hard", "single_hard_dual_better"])
    parser.add_argument("--min_single_steps", type=int, default=40)
    parser.add_argument("--min_step_gain", type=int, default=15)
    parser.add_argument("--png_dpi", type=int, default=360)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--reuse_cached", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = Path(args.save_dir) if args.save_dir else Path(ROOT_DIR) / "log" / "paper_support" / f"relative_slot_heatmaps_{timestamp}"
    ensure_dir(save_dir)

    bay_scan_path = save_dir / "bay_scan.json"
    parallel_scan_path = save_dir / "parallel_scan.json"
    bay_all_path = save_dir / "bay_all.json"
    parallel_all_path = save_dir / "parallel_all.json"
    selected_path = save_dir / "selected_cases.json"

    if args.reuse_cached and bay_scan_path.exists() and parallel_scan_path.exists() and selected_path.exists():
        print("[1/4] reuse cached scan results")
        bay_scan = json.loads(bay_scan_path.read_text())
        parallel_scan = json.loads(parallel_scan_path.read_text())
        selected = json.loads(selected_path.read_text())
        bay_seeds = selected["bay_seeds"]
        parallel_seeds = selected["parallel_seeds"]
    else:
        print("[1/4] scan bay hard/improved cases")
        bay_all = scan_scene_records(args.level, 0, args.forward_ckpt, args.unpark_ckpt, args.max_seed)
        bay_scan = select_case_rows(bay_all, args.selection_mode, args.min_single_steps, args.min_step_gain, args.max_cases_per_type)
        bay_scan_path.write_text(json.dumps(bay_scan, indent=2))
        bay_all_path.write_text(json.dumps(bay_all, indent=2))

        print("[2/4] scan parallel hard/improved cases")
        parallel_all = scan_scene_records(args.level, 1, args.forward_ckpt, args.unpark_ckpt, args.max_seed)
        parallel_scan = select_case_rows(parallel_all, args.selection_mode, args.min_single_steps, args.min_step_gain, args.max_cases_per_type)
        parallel_scan_path.write_text(json.dumps(parallel_scan, indent=2))
        parallel_all_path.write_text(json.dumps(parallel_all, indent=2))

        bay_seeds = [row["seed"] for row in bay_scan]
        parallel_seeds = [row["seed"] for row in parallel_scan]

        selected = {
            "level": args.level,
            "selection_mode": args.selection_mode,
            "min_single_steps": args.min_single_steps,
            "min_step_gain": args.min_step_gain,
            "bay_seeds": bay_seeds,
            "parallel_seeds": parallel_seeds,
            "max_cases_per_type": args.max_cases_per_type,
        }
        selected_path.write_text(json.dumps(selected, indent=2))

    print("[3/4] replay selected cases")
    bay_cases = capture_cases(args.level, 0, bay_seeds, args.forward_ckpt, args.unpark_ckpt)
    parallel_cases = capture_cases(args.level, 1, parallel_seeds, args.forward_ckpt, args.unpark_ckpt)

    print("[4/4] render relative-slot heatmaps")
    plot_slot_relative_heatmaps(args.level, bay_cases, parallel_cases, save_dir, args.png_dpi)

    summary = {
        "save_dir": str(save_dir),
        "figure": "fig07_relative_slot_heatmaps",
        "level": args.level,
        "selection_mode": args.selection_mode,
        "bay_case_count": len(bay_cases),
        "parallel_case_count": len(parallel_cases),
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
