import argparse
import math
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
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


PALETTE = {
    "bg": "#FBFBF8",
    "dark": "#2A3441",
    "muted": "#7A8794",
    "grid": "#E8E3DA",
    "obstacle": "#BFC5CD",
    "ego": "#00976A",
    "slot": "#D35600",
    "reverse": "#E69F00",
    "connector": "#0072B2",
    "anchor": "#F4C363",
    "replay": "#CC79A7",
}

ZH_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def find_zh_font():
    for path in ZH_FONT_CANDIDATES:
        if os.path.exists(path):
            return fm.FontProperties(fname=path)
    return None


ZH_FONT = find_zh_font()


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral"],
            "font.size": 13,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": PALETTE["dark"],
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def zh_text_kwargs():
    if ZH_FONT is not None:
        return {"fontproperties": ZH_FONT}
    return {}


def bezier_curve(p0, p1, p2, p3, num=120):
    t = np.linspace(0.0, 1.0, num)[:, None]
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )


def transform_box(center, yaw, length, width):
    half_l = length * 0.5
    half_w = width * 0.5
    corners = np.array(
        [
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return corners @ rot.T + np.asarray(center, dtype=np.float64)


def rear_axle_center(center, yaw, length):
    cx, cy = center
    offset = np.array([-0.28 * length, 0.0], dtype=np.float64)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return offset @ rot.T + np.array([cx, cy], dtype=np.float64)


def draw_vehicle(ax, center, yaw, length, width, edgecolor, facecolor="white", lw=2.2, alpha=1.0, z=5):
    poly = transform_box(center, yaw, length, width)
    ax.add_patch(
        Polygon(
            poly,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            alpha=alpha,
            zorder=z,
        )
    )
    front = (poly[1] + poly[2]) * 0.5
    tip = front + 0.28 * np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    ax.add_patch(
        FancyArrowPatch(
            posA=front,
            posB=tip,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw * 0.7,
            color=edgecolor,
            zorder=z + 1,
        )
    )
    axle = rear_axle_center(center, yaw, length)
    ax.add_patch(Circle(axle, radius=0.11, facecolor=edgecolor, edgecolor="none", zorder=z + 1))
    return axle


def draw_slot(ax, center, yaw, length, width, edgecolor, lw=2.2, z=3):
    poly = transform_box(center, yaw, length, width)
    ax.add_patch(
        Polygon(
            poly,
            closed=True,
            fill=False,
            edgecolor=edgecolor,
            linewidth=lw,
            linestyle="-",
            zorder=z,
        )
    )


def draw_obstacles(ax):
    obstacles = [
        (-0.2, 2.1, 3.2, 0.65, -2.0),
        (9.1, 4.7, 1.25, 2.25, -18.0),
        (6.3, 8.1, 1.1, 2.6, 18.0),
        (2.7, 7.9, 2.0, 0.85, 28.0),
    ]
    for x, y, w, h, angle in obstacles:
        rect = Rectangle(
            (x, y),
            w,
            h,
            angle=angle,
            facecolor=PALETTE["obstacle"],
            edgecolor="#B5BAC1",
            linewidth=1.0,
            zorder=1,
        )
        ax.add_patch(rect)


def add_path_arrow(ax, points, color, lw, linestyle="-", z=4):
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=lw, linestyle=linestyle, zorder=z)
    start = points[-8]
    end = points[-1]
    ax.add_patch(
        FancyArrowPatch(
            posA=start,
            posB=end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=lw * 0.7,
            color=color,
            linestyle=linestyle,
            zorder=z + 1,
        )
    )


def label_box(fig, xywh, title, body, facecolor, edgecolor, body_color=None):
    if body_color is None:
        body_color = PALETTE["dark"]
    ax = fig.add_axes(xywh)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.015,rounding_size=0.045",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.8,
        )
    )
    text_kwargs = zh_text_kwargs()
    ax.text(
        0.07,
        0.70,
        title,
        fontsize=16,
        color=edgecolor,
        fontweight="bold",
        ha="left",
        va="center",
        **text_kwargs,
    )
    ax.text(
        0.07,
        0.32,
        body,
        fontsize=11.5,
        color=body_color,
        ha="left",
        va="center",
        linespacing=1.45,
        **text_kwargs,
    )
    return ax


def draw_flow_column(fig):
    blocks = [
        (
            [0.73, 0.68, 0.23, 0.16],
            "泊入模型 $\\pi_{in}$",
            "输入当前观测，执行前向探索；\n若局部 RS assist 可用，则直接辅助跟踪。",
            "#E8F6F1",
            PALETTE["ego"],
        ),
        (
            [0.73, 0.47, 0.23, 0.16],
            "泊出模型 $\\pi_{out}$",
            "从目标车位初始化反向 rollout，\n生成一组可候选的反向锚点。",
            "#FFF3DD",
            PALETTE["reverse"],
        ),
        (
            [0.73, 0.26, 0.23, 0.16],
            "RS 连接筛选",
            "当前状态与候选锚点做 RS 可达性判断；\n一旦命中，切换到连接阶段。",
            "#E8F1F8",
            PALETTE["connector"],
        ),
        (
            [0.73, 0.05, 0.23, 0.16],
            "连接后执行",
            "先沿 RS 连接轨迹靠近锚点，\n再反放泊出动作完成最终入位。",
            "#F8EDF6",
            PALETTE["replay"],
        ),
    ]

    axes = []
    for xywh, title, body, facecolor, edgecolor in blocks:
        axes.append(label_box(fig, xywh, title, body, facecolor, edgecolor))

    overlay = fig.add_axes([0, 0, 1, 1], zorder=20)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")

    for idx in range(len(blocks) - 1):
        x = 0.845
        y0 = blocks[idx][0][1]
        y1 = blocks[idx + 1][0][1] + blocks[idx + 1][0][3]
        overlay.add_patch(
            FancyArrowPatch(
                posA=(x, y0 - 0.01),
                posB=(x, y1 + 0.01),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.8,
                color=PALETTE["muted"],
            )
        )


def draw_scene_panel(fig, panel_rect=(0.05, 0.08, 0.62, 0.84), show_text=True, show_legend=True):
    ax = fig.add_axes(panel_rect)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(PALETTE["bg"])

    ax.add_patch(Rectangle((0, 0), 16, 10, facecolor=PALETTE["bg"], edgecolor="none", zorder=0))
    for y in (1.6, 2.3):
        ax.plot([0.0, 16.0], [y, y], color=PALETTE["grid"], linewidth=6.0, zorder=0, alpha=0.75)

    draw_obstacles(ax)

    start_center = (2.5, 1.55)
    start_yaw = math.radians(10.0)
    current_center = (6.55, 3.65)
    current_yaw = math.radians(38.0)
    slot_center = (12.7, 6.85)
    slot_yaw = math.radians(82.0)
    anchor_points = np.array(
        [
            [11.85, 6.25],
            [10.55, 5.55],
            [9.25, 4.95],
            [8.10, 4.55],
        ],
        dtype=np.float64,
    )
    selected_anchor = anchor_points[-1]

    forward_path = bezier_curve(start_center, (3.8, 1.0), (5.2, 2.0), current_center)
    reverse_rollout = bezier_curve(slot_center, (12.0, 6.3), (10.0, 5.2), tuple(anchor_points[-1]))
    replay_path = bezier_curve(tuple(selected_anchor), (9.3, 4.9), (11.2, 5.9), slot_center)
    connector_path = bezier_curve(current_center, (6.9, 4.0), (7.4, 4.35), tuple(selected_anchor), num=80)

    add_path_arrow(ax, forward_path, PALETTE["ego"], lw=3.2, linestyle="-", z=4)
    add_path_arrow(ax, reverse_rollout, PALETTE["reverse"], lw=2.8, linestyle="--", z=3)
    add_path_arrow(ax, connector_path, PALETTE["connector"], lw=3.0, linestyle="-", z=5)
    add_path_arrow(ax, replay_path, PALETTE["replay"], lw=3.0, linestyle="-", z=5)

    for idx, anchor in enumerate(anchor_points, start=1):
        ax.add_patch(
            Circle(
                anchor,
                radius=0.12 if idx < len(anchor_points) else 0.16,
                facecolor=PALETTE["anchor"] if idx < len(anchor_points) else PALETTE["connector"],
                edgecolor="white",
                linewidth=1.0,
                zorder=6,
            )
        )
        if show_text:
            ax.text(
                anchor[0] + 0.12,
                anchor[1] + 0.18,
                f"A{idx}",
                fontsize=10.5,
                color=PALETTE["dark"],
                ha="left",
                va="bottom",
                **zh_text_kwargs(),
            )

    draw_slot(ax, slot_center, slot_yaw, length=2.8, width=1.5, edgecolor=PALETTE["slot"], lw=2.5, z=4)
    draw_vehicle(ax, start_center, start_yaw, 2.45, 1.15, PALETTE["ego"], lw=2.5, z=7)
    draw_vehicle(ax, current_center, current_yaw, 2.45, 1.15, PALETTE["ego"], lw=2.5, alpha=0.95, z=8)
    draw_vehicle(ax, slot_center, slot_yaw, 2.45, 1.15, PALETTE["slot"], facecolor="white", lw=2.5, z=7)

    text_kwargs = zh_text_kwargs()
    if show_text:
        ax.text(1.35, 0.86, "起始状态", color=PALETTE["ego"], fontsize=13, fontweight="bold", **text_kwargs)
        ax.text(5.55, 4.35, "当前状态 $x_t$", color=PALETTE["ego"], fontsize=13, fontweight="bold", **text_kwargs)
        ax.text(11.55, 8.35, "目标车位", color=PALETTE["slot"], fontsize=13, fontweight="bold", **text_kwargs)

        ax.text(
            3.25,
            4.95,
            "前向探索轨迹",
            color=PALETTE["ego"],
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=PALETTE["ego"], linewidth=1.4),
            **text_kwargs,
        )
        ax.text(
            10.15,
            3.25,
            "泊出模型反向锚点",
            color=PALETTE["reverse"],
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=PALETTE["reverse"], linewidth=1.4),
            **text_kwargs,
        )
        ax.text(
            7.55,
            5.55,
            "RS 连接",
            color=PALETTE["connector"],
            fontsize=12.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=PALETTE["connector"], linewidth=1.3),
            **text_kwargs,
        )
        ax.text(
            9.55,
            7.85,
            "反放泊出动作",
            color=PALETTE["replay"],
            fontsize=12.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=PALETTE["replay"], linewidth=1.3),
            **text_kwargs,
        )

    ax.add_patch(
        FancyArrowPatch(
            posA=(6.9, 3.95),
            posB=(7.95, 4.48),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color=PALETTE["connector"],
            zorder=10,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            posA=(9.7, 7.45),
            posB=(10.6, 6.65),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=PALETTE["replay"],
            zorder=10,
        )
    )

    if show_legend:
        legend_y = 0.55
        legend_x = 0.65
        legend_specs = [
            (PALETTE["ego"], "-", "泊入模型前向轨迹"),
            (PALETTE["reverse"], "--", "泊出模型反向 rollout"),
            (PALETTE["connector"], "-", "RS 连接轨迹"),
            (PALETTE["replay"], "-", "反放泊出动作"),
        ]
        for idx, (color, style, text) in enumerate(legend_specs):
            y = legend_y - idx * 0.45
            ax.plot([legend_x, legend_x + 0.95], [y, y], color=color, linestyle=style, linewidth=3.0, zorder=9)
            ax.text(legend_x + 1.10, y, text, color=PALETTE["dark"], fontsize=11.2, va="center", **text_kwargs)

    if show_text:
        ax.text(
            0.25,
            9.55,
            "双模型协作泊车示意图",
            fontsize=18,
            color=PALETTE["dark"],
            fontweight="bold",
            ha="left",
            va="center",
            **text_kwargs,
        )
        ax.text(
            0.25,
            9.02,
            "泊入模型负责前向探索，泊出模型从目标车位反向生成锚点；当前向进展不足时，\n"
            "通过 RS 连接器将当前状态接入可达锚点，并反放泊出动作完成最终入位。",
            fontsize=11.4,
            color=PALETTE["dark"],
            ha="left",
            va="top",
            linespacing=1.45,
            **text_kwargs,
        )
    return ax


def export_schematic(out_dir: Path, stem: str, dpi: int, no_text: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    if no_text:
        fig = plt.figure(figsize=(12.0, 7.0))
        draw_scene_panel(fig, panel_rect=(0.02, 0.04, 0.96, 0.92), show_text=False, show_legend=False)
    else:
        fig = plt.figure(figsize=(13.8, 7.6))
        draw_scene_panel(fig)
        draw_flow_column(fig)

    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    svg_path = out_dir / f"{stem}.svg"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, svg_path


def main():
    parser = argparse.ArgumentParser(description="导出双模型协作泊车示意图。")
    parser.add_argument("--out-dir", type=str, default=str(Path(ROOT_DIR) / "log" / "paper_support"))
    parser.add_argument("--stem", type=str, default="dual_model_cooperation_schematic_cn")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--no-text", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    png_path, pdf_path, svg_path = export_schematic(out_dir, args.stem, args.dpi, no_text=args.no_text)
    print(png_path)
    print(pdf_path)
    print(svg_path)


if __name__ == "__main__":
    main()
