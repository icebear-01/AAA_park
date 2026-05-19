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
from PIL import Image, ImageDraw, ImageFont

import export_dual_parking_paper_gallery as paper_gallery


NON_DLP_SOURCE_ROOT = "src/log/paper_support/dual_segment_gallery_effseg_20260416"
DLP_SOURCE_ROOT = "src/log/paper_support/dlp_example_gallery_top30_paper_focus_24x18/cases"

GROUPS = [
    {
        "key": "normal_bay",
        "title": "Easy Bay",
        "source_dir": f"{NON_DLP_SOURCE_ROOT}/normal_bay",
        "is_dlp": False,
    },
    {
        "key": "normal_parallel",
        "title": "Easy Parallel",
        "source_dir": f"{NON_DLP_SOURCE_ROOT}/normal_parallel",
        "is_dlp": False,
    },
    {
        "key": "complex_bay",
        "title": "Hard Bay",
        "source_dir": f"{NON_DLP_SOURCE_ROOT}/complex_bay",
        "is_dlp": False,
    },
    {
        "key": "complex_parallel",
        "title": "Hard Parallel",
        "source_dir": f"{NON_DLP_SOURCE_ROOT}/complex_parallel",
        "is_dlp": False,
    },
    {
        "key": "extrem_parallel",
        "title": "Extrem Parallel",
        "source_dir": f"{NON_DLP_SOURCE_ROOT}/extrem_parallel",
        "is_dlp": False,
    },
    {
        "key": "dlp",
        "title": "DLP",
        "source_dir": DLP_SOURCE_ROOT,
        "is_dlp": True,
    },
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_style(args):
    paper_gallery.apply_reference_style()
    paper_gallery.vector_plots.DISPLAY_FOOTPRINT_PATH_LINEWIDTH = float(args.path_linewidth)
    paper_gallery.vector_plots.DISPLAY_FOOTPRINT_OUTLINE_LINEWIDTH = float(args.footprint_linewidth)
    paper_gallery.vector_plots.DISPLAY_FOOTPRINT_OUTLINE_FINAL_LINEWIDTH = float(args.final_footprint_linewidth)
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
    return {
        "xmin": float(center_x - 0.5 * float(width)),
        "xmax": float(center_x + 0.5 * float(width)),
        "ymin": float(center_y - 0.5 * float(height)),
        "ymax": float(center_y + 0.5 * float(height)),
    }


def list_candidates(group):
    source_dir = Path(REPO_DIR) / group["source_dir"]
    rows = []
    for summary_path in sorted(source_dir.glob("*_summary.json")):
        summary = load_json(summary_path)
        if not summary.get("planning_success", True):
            continue
        if not summary.get("final_success", True):
            continue
        stem = summary_path.name[: -len("_summary.json")]
        trace_path = source_dir / f"{stem}_trace.json"
        if not trace_path.exists():
            continue
        rows.append(
            {
                "stem": stem,
                "summary_path": summary_path,
                "trace_path": trace_path,
                "summary": summary,
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["summary"].get("motion_segments", 99)),
            int(row["summary"].get("raw_motion_segments", 99)),
            int(row["summary"].get("step_num", 9999)),
            row["stem"],
        )
    )
    return rows


def render_case(row, group, out_dir: Path, args):
    result = load_json(row["trace_path"])
    if group["is_dlp"]:
        bounds = centered_bounds_from_points(result, args.dlp_width, args.dlp_height)
    else:
        bounds = {
            "xmin": float(args.fixed_xmin),
            "xmax": float(args.fixed_xmax),
            "ymin": float(args.fixed_ymin),
            "ymax": float(args.fixed_ymax),
        }

    patched_result = paper_gallery.clone_result_with_bounds(result, bounds)
    stride = paper_gallery.choose_stride(
        len(result["trajectory_states"]),
        min_stride=args.min_footprint_stride,
        target_footprints=args.target_footprints,
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
    ax.set_xlabel("x (m)", fontsize=args.axis_label_fontsize, color=paper_gallery.vector_plots.PALETTE["dark"])
    ax.set_ylabel("y (m)", fontsize=args.axis_label_fontsize, color=paper_gallery.vector_plots.PALETTE["dark"])
    ax.tick_params(labelsize=args.tick_fontsize)
    fig.subplots_adjust(
        left=args.subplot_left,
        right=args.subplot_right,
        bottom=args.subplot_bottom,
        top=args.subplot_top,
    )

    group_dir = out_dir / group["key"]
    group_dir.mkdir(parents=True, exist_ok=True)
    base = group_dir / row["stem"]
    fig.savefig(base.with_suffix(".png"), dpi=args.png_dpi)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)

    summary = dict(row["summary"])
    summary.update(
        {
            "group": group["key"],
            "group_title": group["title"],
            "stem": row["stem"],
            "source_summary": str(row["summary_path"].relative_to(Path(REPO_DIR))),
            "source_trace": str(row["trace_path"].relative_to(Path(REPO_DIR))),
            "png": str(base.with_suffix(".png").relative_to(Path(REPO_DIR))),
            "pdf": str(base.with_suffix(".pdf").relative_to(Path(REPO_DIR))),
            "svg": str(base.with_suffix(".svg").relative_to(Path(REPO_DIR))),
            "plot_bounds": bounds,
            "stride": int(stride),
            "planning_success": bool(result.get("planning_success", summary.get("planning_success", True))),
            "final_success": bool(result.get("final_success", summary.get("final_success", True))),
            "connection_used": bool(result.get("connection_used", summary.get("connection_used", False))),
        }
    )
    with open(group_dir / f"{row['stem']}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def make_group_panel(group, cases, out_dir: Path):
    group_dir = out_dir / group["key"]
    thumb = 720
    label_h = 58
    pad = 30
    width = len(cases) * thumb + (len(cases) + 1) * pad
    height = thumb + label_h + 2 * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    for idx, case in enumerate(cases, start=1):
        image = Image.open(Path(REPO_DIR) / case["png"]).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        x = pad + (idx - 1) * (thumb + pad)
        y = pad
        label = f"{group['title']} #{idx}"
        draw.text((x, y + 10), label, fill=(30, 42, 60), font=font)
        canvas.paste(image, (x, y + label_h))

    panel_png = group_dir / f"{group['key']}_panel.png"
    panel_pdf = group_dir / f"{group['key']}_panel.pdf"
    canvas.save(panel_png)
    canvas.save(panel_pdf, "PDF", resolution=180)
    return {
        "png": str(panel_png.relative_to(Path(REPO_DIR))),
        "pdf": str(panel_pdf.relative_to(Path(REPO_DIR))),
    }


def make_overview_panel(manifest, out_dir: Path):
    thumb = 430
    label_h = 48
    pad = 22
    cols = 5
    rows = len(manifest["groups"])
    width = cols * thumb + (cols + 1) * pad
    height = rows * (thumb + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    for row_idx, group in enumerate(manifest["groups"]):
        for col_idx, case in enumerate(group["cases"]):
            image = Image.open(Path(REPO_DIR) / case["png"]).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
            x = pad + col_idx * (thumb + pad)
            y = pad + row_idx * (thumb + label_h + pad)
            draw.text((x, y + 8), f"{group['title']} #{col_idx + 1}", fill=(30, 42, 60), font=font)
            canvas.paste(image, (x, y + label_h))

    overview_png = out_dir / "overview_pick5_fixed_axes.png"
    overview_pdf = out_dir / "overview_pick5_fixed_axes.pdf"
    canvas.save(overview_png)
    canvas.save(overview_pdf, "PDF", resolution=180)
    return {
        "png": str(overview_png.relative_to(Path(REPO_DIR))),
        "pdf": str(overview_pdf.relative_to(Path(REPO_DIR))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/log/paper_support/planning_success_pick5_fixed_axes",
    )
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--png-dpi", type=int, default=450)
    parser.add_argument("--fixed-xmin", type=float, default=-12.0)
    parser.add_argument("--fixed-xmax", type=float, default=12.0)
    parser.add_argument("--fixed-ymin", type=float, default=-4.0)
    parser.add_argument("--fixed-ymax", type=float, default=16.0)
    parser.add_argument("--dlp-width", type=float, default=24.0)
    parser.add_argument("--dlp-height", type=float, default=20.0)
    parser.add_argument("--min-footprint-stride", type=int, default=5)
    parser.add_argument("--target-footprints", type=int, default=10)
    parser.add_argument("--axis-label-fontsize", type=float, default=14.0)
    parser.add_argument("--tick-fontsize", type=float, default=17.0)
    parser.add_argument("--path-linewidth", type=float, default=3.0)
    parser.add_argument("--footprint-linewidth", type=float, default=1.45)
    parser.add_argument("--final-footprint-linewidth", type=float, default=1.70)
    parser.add_argument("--subplot-left", type=float, default=0.14)
    parser.add_argument("--subplot-right", type=float, default=0.97)
    parser.add_argument("--subplot-bottom", type=float, default=0.12)
    parser.add_argument("--subplot-top", type=float, default=0.97)
    args = parser.parse_args()

    out_dir = Path(REPO_DIR) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style(args)

    manifest = {
        "output_dir": str(out_dir.relative_to(Path(REPO_DIR))),
        "per_group": int(args.per_group),
        "fixed_bounds": {
            "xmin": float(args.fixed_xmin),
            "xmax": float(args.fixed_xmax),
            "ymin": float(args.fixed_ymin),
            "ymax": float(args.fixed_ymax),
        },
        "dlp_bounds_size": {
            "width": float(args.dlp_width),
            "height": float(args.dlp_height),
        },
        "line_widths": {
            "path": float(args.path_linewidth),
            "footprint": float(args.footprint_linewidth),
            "final_footprint": float(args.final_footprint_linewidth),
        },
        "subplot_adjust": {
            "left": float(args.subplot_left),
            "right": float(args.subplot_right),
            "bottom": float(args.subplot_bottom),
            "top": float(args.subplot_top),
        },
        "groups": [],
    }

    for group in GROUPS:
        candidates = list_candidates(group)
        selected = candidates[: args.per_group]
        if len(selected) < args.per_group:
            raise RuntimeError(f"{group['key']} only has {len(selected)} qualified candidates")
        case_summaries = [render_case(row, group, out_dir, args) for row in selected]
        panel = make_group_panel(group, case_summaries, out_dir)
        manifest["groups"].append(
            {
                "key": group["key"],
                "title": group["title"],
                "source_dir": group["source_dir"],
                "panel": panel,
                "cases": case_summaries,
            }
        )

    manifest["overview_panel"] = make_overview_panel(manifest, out_dir)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(out_dir)


if __name__ == "__main__":
    main()
