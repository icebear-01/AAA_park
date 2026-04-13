import argparse
import os
import sys
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for path in [ROOT_DIR, CURRENT_DIR]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

from env.car_parking_out_base import CarParkingOut


PALETTE = {
    "dark": "#223044",
    "accent": "#355070",
    "slot": "#D98F2B",
    "ego": "#2A9D8F",
    "bg": "#FBFBF8",
    "grid": "#D6DADF",
}

ZH_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
ZH_FONT = fm.FontProperties(fname=ZH_FONT_PATH) if os.path.exists(ZH_FONT_PATH) else None


def setup_style(lang: str = "en"):
    rc = {
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.edgecolor": PALETTE["dark"],
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if lang == "zh":
        rc.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"],
                "axes.unicode_minus": False,
            }
        )
    else:
        rc.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            }
        )
    plt.rcParams.update(rc)


def sample_occ_grid(level: str, seed: int):
    np.random.seed(seed)
    env = CarParkingOut(render_mode="rgb_array", fps=0, verbose=False, img_mode="occ_grid")
    try:
        obs = env.reset(level=level)
        img = np.transpose(obs["img"], (2, 0, 1))
        if img.shape != (3, 64, 64):
            raise RuntimeError(f"Unexpected occ_grid shape: {img.shape}")
        return img
    finally:
        env.close()


def make_overlay(channels):
    obstacle, slot, ego = channels
    overlay = np.zeros((64, 64, 3), dtype=np.float32)
    overlay[..., 0] = np.clip(slot * 0.85 + obstacle * 0.75, 0.0, 1.0)
    overlay[..., 1] = np.clip(ego * 0.80, 0.0, 1.0)
    overlay[..., 2] = np.clip(obstacle * 0.90, 0.0, 1.0)
    return overlay


def text_font_kwargs(lang: str):
    if lang == "zh" and ZH_FONT is not None:
        return {"fontproperties": ZH_FONT}
    return {}


def get_labels(lang: str):
    if lang == "zh":
        return {
            "titles": [
                "C0：障碍物占据",
                "C1：原车位区域",
                "C2：自车占据",
                "通道叠加视图",
            ],
            "caption_head": "以自车为中心对齐的局部 occ_grid 输入。",
            "caption_body": "三个通道分别编码周围障碍物、原车位区域与当前自车 footprint，"
            "并在同一 64×64 局部坐标系中栅格化后堆叠。",
            "figure_title": "反向泊出策略的 occ_grid 输入示意图",
            "stack_text": "堆叠后的 occ_grid 输入：3 × 64 × 64",
        }
    return {
        "titles": [
            "C0: Obstacle Occupancy",
            "C1: Slot Mask",
            "C2: Ego Footprint",
            "Overlay View",
        ],
        "caption_head": "Ego-aligned local occupancy grid for unparking.",
        "caption_body": "The input stacks obstacle geometry, original slot region, and the current ego footprint. "
        "Each channel is rasterized in the same local 64x64 frame.",
        "figure_title": "occ_grid Input Representation for Reverse Unparking Policy",
        "stack_text": r"Stacked occ\_grid input: $3 \times 64 \times 64$",
    }


def draw_tensor_stack(ax, labels, lang: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text_kwargs = text_font_kwargs(lang)

    rects = [
        (0.20, 0.20, 0.42, 0.50, PALETTE["accent"], "C0"),
        (0.28, 0.30, 0.42, 0.50, PALETTE["slot"], "C1"),
        (0.36, 0.40, 0.42, 0.50, PALETTE["ego"], "C2"),
    ]
    for x, y, w, h, color, label in rects:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                facecolor=color,
                edgecolor=PALETTE["dark"],
                linewidth=1.1,
                alpha=0.80,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=16,
            fontweight="bold",
            **text_kwargs,
        )

    ax.text(
        0.50,
        0.06,
        labels["stack_text"],
        ha="center",
        va="bottom",
        color=PALETTE["dark"],
        fontsize=12,
        **text_kwargs,
    )


def plot_standard_figure(channels, out_dir: Path, stem: str, labels, lang: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    text_kwargs = text_font_kwargs(lang)

    cmaps = ["gray", "copper", "viridis"]

    fig = plt.figure(figsize=(13.5, 7.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.22], hspace=0.10, wspace=0.18)

    for idx, (title, channel, cmap) in enumerate(zip(labels["titles"][:3], channels, cmaps)):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(channel, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, color=PALETTE["dark"], **text_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["grid"])
            spine.set_linewidth(0.8)

    ax_stack = fig.add_subplot(gs[0, 3])
    ax_stack.imshow(make_overlay(channels), interpolation="nearest")
    ax_stack.set_title(labels["titles"][3], color=PALETTE["dark"], **text_kwargs)
    ax_stack.set_xticks([])
    ax_stack.set_yticks([])
    for spine in ax_stack.spines.values():
        spine.set_color(PALETTE["grid"])
        spine.set_linewidth(0.8)

    ax_caption = fig.add_subplot(gs[1, :3])
    ax_caption.axis("off")
    ax_caption.text(
        0.0,
        0.70,
        labels["caption_head"],
        fontsize=13,
        color=PALETTE["dark"],
        fontweight="semibold",
        **text_kwargs,
    )
    ax_caption.text(
        0.0,
        0.18,
        labels["caption_body"],
        fontsize=11,
        color=PALETTE["dark"],
        **text_kwargs,
    )

    ax_tensor = fig.add_subplot(gs[1, 3])
    draw_tensor_stack(ax_tensor, labels, lang)

    fig.suptitle(labels["figure_title"], fontsize=15, color=PALETTE["dark"], y=0.98, **text_kwargs)

    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_double_column_figure(channels, out_dir: Path, stem: str, labels, lang: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmaps = ["gray", "copper", "viridis"]
    text_kwargs = text_font_kwargs(lang)

    fig = plt.figure(figsize=(7.2, 3.25))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.24], hspace=0.08, wspace=0.12)

    for idx, (title, channel, cmap) in enumerate(zip(labels["titles"][:3], channels, cmaps)):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(channel, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, color=PALETTE["dark"], fontsize=10.5, pad=4, **text_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["grid"])
            spine.set_linewidth(0.75)

    ax_overlay = fig.add_subplot(gs[0, 3])
    ax_overlay.imshow(make_overlay(channels), interpolation="nearest")
    ax_overlay.set_title(labels["titles"][3], color=PALETTE["dark"], fontsize=10.5, pad=4, **text_kwargs)
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])
    for spine in ax_overlay.spines.values():
        spine.set_color(PALETTE["grid"])
        spine.set_linewidth(0.75)

    ax_caption = fig.add_subplot(gs[1, :3])
    ax_caption.axis("off")
    ax_caption.text(
        0.0,
        0.68,
        labels["caption_head"],
        fontsize=10.5,
        color=PALETTE["dark"],
        fontweight="semibold",
        **text_kwargs,
    )
    ax_caption.text(
        0.0,
        0.10,
        labels["caption_body"],
        fontsize=9.3,
        color=PALETTE["dark"],
        **text_kwargs,
    )

    ax_tensor = fig.add_subplot(gs[1, 3])
    draw_tensor_stack(ax_tensor, labels, lang)

    fig.suptitle(labels["figure_title"], fontsize=12.5, color=PALETTE["dark"], y=0.99, **text_kwargs)

    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="Extrem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="src/log/paper_support")
    parser.add_argument("--stem", type=str, default="occ_grid_input_schematic")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--layout", type=str, default="standard", choices=["standard", "double-column"])
    args = parser.parse_args()

    setup_style(args.lang)
    channels = sample_occ_grid(level=args.level, seed=args.seed)
    labels = get_labels(args.lang)
    if args.layout == "double-column":
        png_path, pdf_path = plot_double_column_figure(channels, Path(args.output_dir), args.stem, labels, args.lang)
    else:
        png_path, pdf_path = plot_standard_figure(channels, Path(args.output_dir), args.stem, labels, args.lang)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
