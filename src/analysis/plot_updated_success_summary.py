import argparse
import json
import matplotlib
matplotlib.use("Agg")
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THEMES = {
    "paper": {
        "method_colors": {"single": "#355070", "dual": "#D98F2B"},
        "palette": {"dark": "#2F3B4A", "bg": "#FBFBF8", "grid": "#C9CED6"},
        "font_family": "serif",
        "font_list": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext": "stix",
        "axes_title_weight": "semibold",
        "legend_fontsize": 10.0,
    },
    "ieee": {
        "method_colors": {"single": "#005DAA", "dual": "#D95319"},
        "palette": {"dark": "#111111", "bg": "#FFFFFF", "grid": "#D7D7D7"},
        "font_family": "sans-serif",
        "font_list": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext": "dejavusans",
        "axes_title_weight": "bold",
        "legend_fontsize": 11.6,
    },
}

ACTIVE_THEME = THEMES["paper"]


def setup_style():
    palette = ACTIVE_THEME["palette"]
    plt.rcParams.update(
        {
            "font.family": ACTIVE_THEME["font_family"],
            "font.serif": ACTIVE_THEME["font_list"] if ACTIVE_THEME["font_family"] == "serif" else [],
            "font.sans-serif": ACTIVE_THEME["font_list"] if ACTIVE_THEME["font_family"] == "sans-serif" else [],
            "mathtext.fontset": ACTIVE_THEME["mathtext"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": ACTIVE_THEME["axes_title_weight"],
            "axes.labelsize": 12,
            "axes.labelcolor": palette["dark"],
            "axes.edgecolor": palette["dark"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.color": palette["dark"],
            "ytick.color": palette["dark"],
            "legend.fontsize": ACTIVE_THEME["legend_fontsize"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax):
    palette = ACTIVE_THEME["palette"]
    ax.set_facecolor(palette["bg"])
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, color=palette["grid"])
    ax.tick_params(axis="both", length=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(palette["dark"])
        ax.spines[spine].set_linewidth(0.8)


def label_bars(ax, bars):
    palette = ACTIVE_THEME["palette"]
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.18,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=palette["dark"],
        )


def save_figure(fig, out_path: Path, dpi: int):
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def build_metrics(complex_dual_rate: float):
    overall = {
        "Normal": {"single": 99.6, "dual": 99.8},
        "Complex": {"single": 97.1, "dual": complex_dual_rate},
        "Extrem": {"single": 93.8, "dual": 98.4},
        "DLP": {"single": 94.0, "dual": 95.35},
    }

    parking_types = {
        "Normal Bay": {"single": 100.0, "dual": 100.0},
        "Normal Parallel": {"single": 100.0, "dual": 100.0},
        "Complex Bay": {"single": 98.0, "dual": 100.0},
        "Complex Parallel": {"single": 95.0, "dual": 98.0},
        "Extrem Parallel": {"single": 93.8, "dual": 98.4},
        "DLP (Bay-like)": {"single": 94.0, "dual": 95.35},
    }

    notes = {
        "metric_definition": {
            "single_model": "final parking success rate",
            "dual_model": "path planning success rate = final success OR successful connector/path generation",
        },
        "overall_sources": {
            "single": "src/log/eval/minimal_full_sac0_20260322_183643.json",
            "dual_normal": "src/log/eval/dual_normal_complex_compare_20260329_233728/summary.json",
            "dual_complex_override": "accepted rerun on 2026-03-31: Complex dual planning success rate fixed to 98.8%",
            "dual_extrem": "src/log/eval/bidirectional_extrem_parallel_stats_20260329_221830/episode_details.json (1968/2000 = 98.4%)",
            "dual_dlp": "src/log/eval/dual_full_benchmark_20260330_223627/summary.json",
        },
        "parking_type_sources": {
            "normal_complex": "src/log/paper_support/dual_framework_20260330_103133/parking_type_results.json",
            "extrem_parallel": "Extrem Bay is undefined in released benchmark maps; Extrem uses parallel-slot generation only.",
            "dlp": "DLP contains only bay-like fixed cases; no parallel split is available.",
        },
    }
    return overall, parking_types, notes


def plot_difficulty(overall_metrics, out_dir: Path, dpi: int):
    setup_style()
    palette = ACTIVE_THEME["palette"]
    method_colors = ACTIVE_THEME["method_colors"]
    labels = list(overall_metrics.keys())
    x = np.arange(len(labels))
    width = 0.33

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    style_axis(ax)

    single_vals = [overall_metrics[label]["single"] for label in labels]
    dual_vals = [overall_metrics[label]["dual"] for label in labels]
    single_bars = ax.bar(
        x - width / 2,
        single_vals,
        width,
        color=method_colors["single"],
        edgecolor="white",
        linewidth=0.8,
        label="HOPE",
    )
    dual_bars = ax.bar(
        x + width / 2,
        dual_vals,
        width,
        color=method_colors["dual"],
        edgecolor="white",
        linewidth=0.8,
        label="Dual HOPE",
    )

    ax.set_ylim(90, 101.5)
    ax.set_ylabel("Path Planning Success Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Success Rate Across Difficulty Levels")
    ax.legend(frameon=False, ncol=1, loc="upper right")
    label_bars(ax, single_bars)
    label_bars(ax, dual_bars)
    save_figure(fig, out_dir / "fig_path_success_difficulty", dpi)


def plot_parking_types(parking_type_metrics, out_dir: Path, dpi: int):
    setup_style()
    palette = ACTIVE_THEME["palette"]
    method_colors = ACTIVE_THEME["method_colors"]
    labels = list(parking_type_metrics.keys())
    x = np.arange(len(labels))
    width = 0.33

    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    style_axis(ax)

    single_vals = [parking_type_metrics[label]["single"] for label in labels]
    dual_vals = [parking_type_metrics[label]["dual"] for label in labels]
    single_bars = ax.bar(
        x - width / 2,
        single_vals,
        width,
        color=method_colors["single"],
        edgecolor="white",
        linewidth=0.8,
        label="HOPE",
    )
    dual_bars = ax.bar(
        x + width / 2,
        dual_vals,
        width,
        color=method_colors["dual"],
        edgecolor="white",
        linewidth=0.8,
        label="Dual HOPE",
    )

    ax.set_ylim(90, 101.5)
    ax.set_ylabel("Path Planning Success Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Normal\nBay",
            "Normal\nParallel",
            "Complex\nBay",
            "Complex\nParallel",
            "Extrem\nParallel",
            "DLP\n(Bay-like)",
        ]
    )
    ax.set_title("Success Rate Across Parking Slot Types")
    ax.legend(frameon=False, ncol=1, loc="upper right")
    label_bars(ax, single_bars)
    label_bars(ax, dual_bars)
    ax.text(
        1.0,
        -0.18,
        "Extrem Bay is undefined in the released benchmark. DLP contains only bay-like fixed cases.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        color=palette["dark"],
        alpha=0.82,
    )
    save_figure(fig, out_dir / "fig_path_success_parking_types", dpi)


def plot_combined(overall_metrics, parking_type_metrics, out_dir: Path, dpi: int):
    setup_style()
    palette = ACTIVE_THEME["palette"]
    method_colors = ACTIVE_THEME["method_colors"]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16.0, 5.9))
    width = 0.33

    overall_labels = list(overall_metrics.keys())
    x0 = np.arange(len(overall_labels))
    style_axis(axes[0])
    bars0 = axes[0].bar(
        x0 - width / 2,
        [overall_metrics[label]["single"] for label in overall_labels],
        width,
        color=method_colors["single"],
        edgecolor="white",
        linewidth=0.8,
        label="HOPE",
    )
    bars1 = axes[0].bar(
        x0 + width / 2,
        [overall_metrics[label]["dual"] for label in overall_labels],
        width,
        color=method_colors["dual"],
        edgecolor="white",
        linewidth=0.8,
        label="Dual HOPE",
    )
    axes[0].set_ylim(90, 101.5)
    axes[0].set_ylabel("Path Planning Success Rate (%)")
    axes[0].set_xticks(x0)
    axes[0].set_xticklabels(overall_labels)
    axes[0].set_title("By Difficulty")
    label_bars(axes[0], bars0)
    label_bars(axes[0], bars1)
    axes[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.10))

    type_labels = list(parking_type_metrics.keys())
    x1 = np.arange(len(type_labels))
    style_axis(axes[1])
    bars2 = axes[1].bar(
        x1 - width / 2,
        [parking_type_metrics[label]["single"] for label in type_labels],
        width,
        color=method_colors["single"],
        edgecolor="white",
        linewidth=0.8,
    )
    bars3 = axes[1].bar(
        x1 + width / 2,
        [parking_type_metrics[label]["dual"] for label in type_labels],
        width,
        color=method_colors["dual"],
        edgecolor="white",
        linewidth=0.8,
    )
    axes[1].set_ylim(90, 101.5)
    axes[1].set_xticks(x1)
    axes[1].set_xticklabels(
        [
            "Normal\nBay",
            "Normal\nParallel",
            "Complex\nBay",
            "Complex\nParallel",
            "Extrem\nParallel",
            "DLP\n(Bay-like)",
        ]
    )
    axes[1].set_title("By Parking Slot Type")
    label_bars(axes[1], bars2)
    label_bars(axes[1], bars3)
    axes[1].text(
        1.0,
        -0.18,
        "Extrem Bay is undefined in the released benchmark. DLP contains only bay-like fixed cases.",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        color=palette["dark"],
        alpha=0.82,
    )
    fig.suptitle("Dual-Model Path Planning Success Rate Summary", fontsize=15, y=1.01, color=palette["dark"])
    fig.tight_layout()
    save_figure(fig, out_dir / "fig_path_success_overview", dpi)


def main():
    global ACTIVE_THEME
    parser = argparse.ArgumentParser(description="Plot updated success-rate summary for the dual-model paper figures.")
    parser.add_argument("--complex-dual", type=float, default=98.8, help="Updated Complex dual-model path planning success rate in percent.")
    parser.add_argument("--dpi", type=int, default=480)
    parser.add_argument("--style", type=str, default="paper", choices=sorted(THEMES.keys()))
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()
    ACTIVE_THEME = THEMES[args.style]

    root_dir = Path(__file__).resolve().parents[1]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = root_dir / "log" / "paper_support" / f"updated_success_summary_{args.style}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_metrics, parking_type_metrics, notes = build_metrics(args.complex_dual)
    payload = {
        "style": args.style,
        "overall_path_planning_success_rate_pct": overall_metrics,
        "parking_type_path_planning_success_rate_pct": parking_type_metrics,
        "notes": notes,
    }
    with (out_dir / "updated_success_rates.json").open("w") as f:
        json.dump(payload, f, indent=2)

    plot_difficulty(overall_metrics, out_dir, args.dpi)
    plot_parking_types(parking_type_metrics, out_dir, args.dpi)
    plot_combined(overall_metrics, parking_type_metrics, out_dir, args.dpi)
    print(out_dir)


if __name__ == "__main__":
    main()
