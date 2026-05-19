import argparse
import json
import os
import sys
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

import export_dual_parking_paper_gallery as paper_gallery


DEFAULT_CASES = [
    {
        "label_cn": "Easy Bay",
        "basename": "normal_bay_seed2858",
        "trace": "src/log/paper_support/selected_planning_success_cases/normal_bay/normal_bay_seed2858_trace.json",
    },
    {
        "label_cn": "Easy Parallel",
        "basename": "normal_parallel_seed1393",
        "trace": "src/log/paper_support/selected_planning_success_cases/normal_parallel/normal_parallel_seed1393_trace.json",
    },
    {
        "label_cn": "Hard Bay",
        "basename": "complex_bay_seed1139",
        "trace": "src/log/paper_support/selected_planning_success_cases/complex_bay/complex_bay_seed1139_trace.json",
    },
    {
        "label_cn": "Hard Parallel",
        "basename": "complex_parallel_seed1362",
        "trace": "src/log/paper_support/selected_planning_success_cases/complex_parallel/complex_parallel_seed1362_trace.json",
    },
    {
        "label_cn": "Extrem Parallel",
        "basename": "extrem_parallel_seed2932",
        "trace": "src/log/paper_support/selected_planning_success_cases/extrem_parallel/extrem_parallel_seed2932_trace.json",
    },
    {
        "label_cn": "DLP",
        "basename": "dlp_case018_seed1",
        "trace": "src/log/paper_support/dlp_example_gallery_top30_paper_focus_24x18/cases/dlp_case018_seed1_trace.json",
    },
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def square_bounds(bounds, extra_margin_ratio=0.0):
    xmin, xmax = float(bounds["xmin"]), float(bounds["xmax"])
    ymin, ymax = float(bounds["ymin"]), float(bounds["ymax"])
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin)
    span *= 1.0 + float(extra_margin_ratio)
    half = 0.5 * max(span, 1.0)
    return {
        "xmin": float(center_x - half),
        "xmax": float(center_x + half),
        "ymin": float(center_y - half),
        "ymax": float(center_y + half),
    }


def collect_focus_points(result):
    points = []
    for state in result.get("trajectory_states", []):
        points.append((float(state["x"]), float(state["y"])))

    map_payload = result.get("map", {})
    for key in ("start_box", "dest_box"):
        for point in map_payload.get(key, []):
            points.append((float(point[0]), float(point[1])))

    connector = result.get("connector_path")
    if connector is not None:
        points.extend(zip(connector.get("x", []), connector.get("y", [])))

    return np.asarray(points, dtype=np.float64)


def centered_bounds_from_points(result, width, height):
    points = collect_focus_points(result)
    if points.size == 0:
        bounds = result["map"]["bounds"]
        center_x = 0.5 * (float(bounds["xmin"]) + float(bounds["xmax"]))
        center_y = 0.5 * (float(bounds["ymin"]) + float(bounds["ymax"]))
    else:
        center_x = 0.5 * (float(np.min(points[:, 0])) + float(np.max(points[:, 0])))
        center_y = 0.5 * (float(np.min(points[:, 1])) + float(np.max(points[:, 1])))
    half_w = 0.5 * float(width)
    half_h = 0.5 * float(height)
    return {
        "xmin": float(center_x - half_w),
        "xmax": float(center_x + half_w),
        "ymin": float(center_y - half_h),
        "ymax": float(center_y + half_h),
    }


def fixed_or_square_bounds(result, case_spec, fixed_bounds, dlp_width, dlp_height, margin_ratio):
    if case_spec["basename"].startswith("dlp_"):
        bounds = centered_bounds_from_points(result, dlp_width, dlp_height)
        if margin_ratio <= 0:
            return bounds
        center_x = 0.5 * (bounds["xmin"] + bounds["xmax"])
        center_y = 0.5 * (bounds["ymin"] + bounds["ymax"])
        width = (bounds["xmax"] - bounds["xmin"]) * (1.0 + float(margin_ratio))
        height = (bounds["ymax"] - bounds["ymin"]) * (1.0 + float(margin_ratio))
        return {
            "xmin": float(center_x - 0.5 * width),
            "xmax": float(center_x + 0.5 * width),
            "ymin": float(center_y - 0.5 * height),
            "ymax": float(center_y + 0.5 * height),
        }
    return dict(fixed_bounds)


def apply_style():
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


def render_case(case_spec, out_dir: Path, png_dpi: int, margin_ratio: float, fixed_bounds, dlp_width, dlp_height):
    trace_path = Path(REPO_DIR) / case_spec["trace"]
    result = load_json(trace_path)
    bounds = fixed_or_square_bounds(result, case_spec, fixed_bounds, dlp_width, dlp_height, margin_ratio)
    patched_result = paper_gallery.clone_result_with_bounds(result, bounds)
    stride = paper_gallery.choose_stride(
        len(result["trajectory_states"]),
        min_stride=5,
        target_footprints=10,
    )

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    paper_gallery.vector_plots.draw_case(
        ax,
        "",
        patched_result,
        show_legend=False,
        show_axis_labels=True,
        trajectory_style="footprints",
        footprint_stride=stride,
    )
    ax.set_title("")
    ax.set_xlabel("x (m)", fontsize=14.0, color=paper_gallery.vector_plots.PALETTE["dark"])
    ax.set_ylabel("y (m)", fontsize=14.0, color=paper_gallery.vector_plots.PALETTE["dark"])
    ax.tick_params(labelsize=17.0)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.12, top=0.97)

    base = out_dir / case_spec["basename"]
    fig.savefig(base.with_suffix(".png"), dpi=png_dpi)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)

    return {
        "label_cn": case_spec["label_cn"],
        "basename": case_spec["basename"],
        "source_trace": case_spec["trace"],
        "png": str(base.with_suffix(".png").relative_to(Path(REPO_DIR))),
        "pdf": str(base.with_suffix(".pdf").relative_to(Path(REPO_DIR))),
        "svg": str(base.with_suffix(".svg").relative_to(Path(REPO_DIR))),
        "square_bounds": bounds,
        "step_num": result.get("step_num"),
        "planning_success": result.get("planning_success"),
        "final_success": result.get("final_success"),
        "connection_used": result.get("connection_used"),
        "stride": int(stride),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/log/paper_support/selected_planning_success_cases/curated_selection_square",
    )
    parser.add_argument("--png-dpi", type=int, default=450)
    parser.add_argument("--margin-ratio", type=float, default=0.0)
    parser.add_argument("--fixed-xmin", type=float, default=-12.0)
    parser.add_argument("--fixed-xmax", type=float, default=12.0)
    parser.add_argument("--fixed-ymin", type=float, default=-4.0)
    parser.add_argument("--fixed-ymax", type=float, default=16.0)
    parser.add_argument("--dlp-width", type=float, default=24.0)
    parser.add_argument("--dlp-height", type=float, default=20.0)
    args = parser.parse_args()

    out_dir = Path(REPO_DIR) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    fixed_bounds = {
        "xmin": float(args.fixed_xmin),
        "xmax": float(args.fixed_xmax),
        "ymin": float(args.fixed_ymin),
        "ymax": float(args.fixed_ymax),
    }
    manifest = {
        "output_dir": str(out_dir.relative_to(Path(REPO_DIR))),
        "fixed_bounds": fixed_bounds,
        "dlp_bounds_size": {
            "width": float(args.dlp_width),
            "height": float(args.dlp_height),
        },
        "margin_ratio": float(args.margin_ratio),
        "cases": [
            render_case(
                case_spec,
                out_dir,
                args.png_dpi,
                args.margin_ratio,
                fixed_bounds,
                args.dlp_width,
                args.dlp_height,
            )
            for case_spec in DEFAULT_CASES
        ],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(out_dir)


if __name__ == "__main__":
    main()
