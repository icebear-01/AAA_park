import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageFilter


SCENES = ["Normal", "Complex", "Extrem"]
METHOD_LABELS = ["HOPE final", "Ours planning", "Ours final"]
PALETTE = {
    "navy": "#355070",
    "orange": "#D98F2B",
    "teal": "#2A9D8F",
    "red": "#C8553D",
    "gray": "#B7BDC5",
    "dark": "#273043",
    "offwhite": "#FBFBF8",
    "grid": "#C9CED6",
}


def setup_publication_style():
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
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.color": PALETTE["dark"],
            "ytick.color": PALETTE["dark"],
            "legend.fontsize": 10.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, y_grid=True):
    ax.set_facecolor(PALETTE["offwhite"])
    if y_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, color=PALETTE["grid"])
    ax.tick_params(axis="both", length=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["dark"])
        ax.spines[spine].set_linewidth(0.8)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def scene_title(scene: str) -> str:
    return {"Normal": "Simple", "Complex": "Complex", "Extrem": "Extreme"}.get(scene, scene)


def collect_metrics(single_summary, dual_nc_summary, dual_ext_summary, dual_ext_details):
    metrics = {}

    for scene in ("Normal", "Complex"):
        single_rate = single_summary["levels"][scene]["success_rate"]
        dual_scene = dual_nc_summary["results"][scene]
        metrics[scene] = {
            "single_final_rate": single_rate,
            "ours_planning_rate": dual_scene["path_planning_success_rate"],
            "ours_final_rate": dual_scene["final_parking_success_rate"],
        }

    ext_single = single_summary["levels"]["Extrem"]["success_rate"]
    ext_direct = sum(
        int((not item["connection_used"]) and item["final_parking_success"])
        for item in dual_ext_details
    )
    ext_connected_parked = sum(
        int(item["connection_used"] and item["final_parking_success"])
        for item in dual_ext_details
    )
    ext_connected_fail = sum(
        int(item["connection_used"] and (not item["final_parking_success"]))
        for item in dual_ext_details
    )
    ext_fail = sum(
        int((not item["connection_used"]) and (not item["final_parking_success"]))
        for item in dual_ext_details
    )
    ext_total = len(dual_ext_details)
    ext_planning_rate = (ext_direct + ext_connected_parked + ext_connected_fail) / ext_total
    ext_final_rate = (ext_direct + ext_connected_parked) / ext_total
    metrics["Extrem"] = {
        "single_final_rate": ext_single,
        "ours_planning_rate": ext_planning_rate,
        "ours_final_rate": ext_final_rate,
        "breakdown_counts": {
            "direct_park": ext_direct,
            "connected_parked": ext_connected_parked,
            "connected_exec_fail": ext_connected_fail,
            "fail": ext_fail,
        },
    }

    for scene in ("Normal", "Complex"):
        details = dual_nc_summary["details"][scene]
        direct_park = sum(
            int((not item["connection_success"]) and item["final_success"]) for item in details
        )
        connected_parked = sum(
            int(item["connection_success"] and item["final_success"]) for item in details
        )
        connected_exec_fail = sum(
            int(item["connection_success"] and (not item["final_success"])) for item in details
        )
        fail = sum(
            int((not item["connection_success"]) and (not item["final_success"])) for item in details
        )
        metrics[scene]["breakdown_counts"] = {
            "direct_park": direct_park,
            "connected_parked": connected_parked,
            "connected_exec_fail": connected_exec_fail,
            "fail": fail,
        }

    return metrics


def load_dual_nc_summary(summary_path: Path, details_path: Path):
    summary = load_json(summary_path)
    details = load_json(details_path)
    summary["details"] = details
    return summary


def plot_grouped_results(metrics, out_dir: Path):
    setup_publication_style()
    scenes = SCENES
    x = np.arange(len(scenes))
    width = 0.24
    colors = [PALETTE["navy"], PALETTE["orange"], PALETTE["teal"]]

    single_vals = [metrics[s]["single_final_rate"] * 100 for s in scenes]
    ours_plan_vals = [metrics[s]["ours_planning_rate"] * 100 for s in scenes]
    ours_final_vals = [metrics[s]["ours_final_rate"] * 100 for s in scenes]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    style_axis(ax)
    bars1 = ax.bar(
        x - width,
        single_vals,
        width,
        label=METHOD_LABELS[0],
        color=colors[0],
        edgecolor="white",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        x,
        ours_plan_vals,
        width,
        label=METHOD_LABELS[1],
        color=colors[1],
        edgecolor="white",
        linewidth=0.8,
    )
    bars3 = ax.bar(
        x + width,
        ours_final_vals,
        width,
        label=METHOD_LABELS[2],
        color=colors[2],
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_ylim(70, 101.5)
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([scene_title(s) for s in scenes])
    ax.set_title("Planning and Execution Success Across Parking Scenarios")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.text(
        0.01,
        1.02,
        "Higher is better",
        transform=ax.transAxes,
        fontsize=10.5,
        color=PALETTE["dark"],
        alpha=0.8,
    )

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=PALETTE["dark"],
            )

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"paper_main_results.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_breakdown(metrics, out_dir: Path):
    setup_publication_style()
    categories = [
        ("direct_park", "Parked without connection", PALETTE["navy"]),
        ("connected_parked", "Connected and parked", PALETTE["teal"]),
        ("connected_exec_fail", "Connected but execution failed", PALETTE["red"]),
        ("fail", "No plan / no park", PALETTE["gray"]),
    ]

    x = np.arange(len(SCENES))
    width = 0.62
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    style_axis(ax)
    bottoms = np.zeros(len(SCENES), dtype=float)
    totals = [sum(metrics[s]["breakdown_counts"].values()) for s in SCENES]

    for key, label, color in categories:
        vals = np.array([metrics[s]["breakdown_counts"][key] / totals[i] * 100 for i, s in enumerate(SCENES)])
        ax.bar(
            x,
            vals,
            width,
            bottom=bottoms,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        centers = bottoms + vals / 2
        for xi, yi, v in zip(x, centers, vals):
            if v >= 6:
                ax.text(xi, yi, f"{v:.1f}", ha="center", va="center", fontsize=9.5, color="white")
        bottoms += vals

    ax.set_ylim(0, 100)
    ax.set_ylabel("Episode Share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([scene_title(s) for s in SCENES])
    ax.set_title("Outcome Breakdown of the Dual-Model Pipeline")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
    ax.text(
        0.01,
        1.02,
        "Each bar sums to 100% of episodes",
        transform=ax.transAxes,
        fontsize=10.5,
        color=PALETTE["dark"],
        alpha=0.8,
    )
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"paper_dual_breakdown.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(metrics, out_dir: Path):
    setup_publication_style()
    matrix = np.array(
        [
            [metrics[s]["single_final_rate"] * 100 for s in SCENES],
            [metrics[s]["ours_planning_rate"] * 100 for s in SCENES],
            [metrics[s]["ours_final_rate"] * 100 for s in SCENES],
        ]
    )

    cmap = LinearSegmentedColormap.from_list(
        "paper_success",
        ["#F7F4EA", "#BDDCCF", "#5EA8A2", "#355070"],
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_facecolor(PALETTE["offwhite"])
    im = ax.imshow(matrix, cmap=cmap, vmin=80, vmax=100)
    ax.set_xticks(np.arange(len(SCENES)))
    ax.set_xticklabels([scene_title(s) for s in SCENES])
    ax.set_yticks(np.arange(len(METHOD_LABELS)))
    ax.set_yticklabels(METHOD_LABELS)
    ax.set_title("Success-Rate Heatmap")
    ax.tick_params(axis="both", length=0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=PALETTE["dark"], fontsize=10.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success Rate (%)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"paper_success_heatmap.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_case_summary(base_dir: Path, case_id: int, variant: str):
    return load_json(base_dir / f"case_{case_id}" / f"{variant}_summary.json")


def load_smoothed_image(path: Path):
    img = Image.open(path).convert("RGB")
    img = img.resize(
        (int(img.width * 1.15), int(img.height * 1.15)),
        resample=Image.Resampling.LANCZOS,
    )
    base = img.filter(ImageFilter.GaussianBlur(radius=0.25))

    arr = np.asarray(base).astype(np.int16)
    sat = arr.max(axis=2) - arr.min(axis=2)
    val = arr.max(axis=2)
    # Focus on the rendered trajectory colors instead of the whole map.
    route_mask = ((sat > 55) & (val > 105)).astype(np.uint8) * 255

    alpha = Image.fromarray(route_mask, mode="L")
    alpha = alpha.filter(ImageFilter.MaxFilter(5))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.0))

    route_layer = img.filter(ImageFilter.GaussianBlur(radius=0.55))
    route_layer = route_layer.filter(ImageFilter.UnsharpMask(radius=1.4, percent=110, threshold=2))
    route_layer.putalpha(alpha)

    composited = base.convert("RGBA")
    composited.alpha_composite(route_layer)
    composited = composited.convert("RGB")
    composited = composited.filter(ImageFilter.UnsharpMask(radius=1.0, percent=70, threshold=3))
    return np.asarray(composited)


def plot_trajectory_panel(case_viz_dir: Path, out_dir: Path):
    setup_publication_style()
    case_ids = [130, 174]
    variants = [("single", "HOPE"), ("double", "Ours")]
    row_labels = {
        130: "Rescue case: single fails, dual succeeds",
        174: "Failure case: single succeeds, dual over-takes too early",
    }

    fig, axes = plt.subplots(len(case_ids), len(variants), figsize=(13.0, 9.8))

    for row, case_id in enumerate(case_ids):
        for col, (variant, label) in enumerate(variants):
            ax = axes[row, col]
            img_path = case_viz_dir / f"case_{case_id}" / f"{variant}_world.png"
            summary = load_case_summary(case_viz_dir, case_id, variant)
            img = load_smoothed_image(img_path)
            ax.imshow(img, interpolation="lanczos", resample=True)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor(PALETTE["dark"])

            if variant == "single":
                subtitle = f"{label} | {summary['status']} | steps={summary['step_num']}"
            else:
                conn = summary.get("connection_index")
                conn_text = f" | anchor={conn}" if conn is not None else ""
                subtitle = f"{label} | {summary['status']} | steps={summary['step_num']}{conn_text}"

            ax.set_title(subtitle, fontsize=10.5, color=PALETTE["dark"], pad=8)

        fig.text(
            0.03,
            0.74 if row == 0 else 0.30,
            row_labels[case_id],
            rotation=90,
            va="center",
            ha="center",
            fontsize=11.5,
            color=PALETTE["dark"],
        )

    fig.suptitle(
        "Representative Planning Trajectories",
        fontsize=15,
        y=0.98,
        color=PALETTE["dark"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"paper_trajectory_panel.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-summary",
        type=Path,
        default=Path("/home/wmd/AAA-progect/hope_origin/HOPE/src/log/eval/minimal_full_sac0_20260322_183643.json"),
    )
    parser.add_argument(
        "--dual-nc-summary",
        type=Path,
        default=Path("/home/wmd/AAA-progect/hope_origin/HOPE/src/log/eval/dual_normal_complex_compare_20260329_233728/summary.json"),
    )
    parser.add_argument(
        "--dual-nc-details",
        type=Path,
        default=Path("/home/wmd/AAA-progect/hope_origin/HOPE/src/log/eval/dual_normal_complex_compare_20260329_233728/episode_details.json"),
    )
    parser.add_argument(
        "--dual-ext-summary",
        type=Path,
        default=Path("/home/wmd/AAA-progect/hope_origin/HOPE/src/log/eval/bidirectional_extrem_parallel_stats_20260329_221830/summary.json"),
    )
    parser.add_argument(
        "--dual-ext-details",
        type=Path,
        default=Path("/home/wmd/AAA-progect/hope_origin/HOPE/src/log/eval/bidirectional_extrem_parallel_stats_20260329_221830/episode_details.json"),
    )
    parser.add_argument(
        "--case-viz-dir",
        type=Path,
        default=Path("/home/wmd/AAA-progect/hope_origin/HOPE/src/log/analysis/extrem_case_compare_viz_20260329_171815"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(f"/home/wmd/AAA-progect/hope_origin/HOPE/src/log/paper_figs_{stamp}")

    ensure_dir(args.output_dir)

    single_summary = load_json(args.single_summary)
    dual_nc_summary = load_dual_nc_summary(args.dual_nc_summary, args.dual_nc_details)
    dual_ext_summary = load_json(args.dual_ext_summary)
    dual_ext_details = load_json(args.dual_ext_details)
    metrics = collect_metrics(single_summary, dual_nc_summary, dual_ext_summary, dual_ext_details)

    plot_grouped_results(metrics, args.output_dir)
    plot_breakdown(metrics, args.output_dir)
    plot_heatmap(metrics, args.output_dir)
    plot_trajectory_panel(args.case_viz_dir, args.output_dir)

    payload = {
        "single_summary": str(args.single_summary),
        "dual_nc_summary": str(args.dual_nc_summary),
        "dual_ext_summary": str(args.dual_ext_summary),
        "case_viz_dir": str(args.case_viz_dir),
        "metrics": metrics,
        "figures": [
            "paper_main_results.png",
            "paper_dual_breakdown.png",
            "paper_success_heatmap.png",
            "paper_trajectory_panel.png",
        ],
    }
    with (args.output_dir / "metrics_summary.json").open("w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"output_dir": str(args.output_dir), **payload}, indent=2))


if __name__ == "__main__":
    main()
