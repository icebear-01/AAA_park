# -*- coding: utf-8 -*-
import argparse
import json
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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch
from shapely.geometry import Polygon

from configs import (
    UNPARK_IOU_STAGE_THRESHOLDS,
    UNPARK_SUCCESS_IOU,
    VehicleBox,
)
from env.task_utils import calc_iou


PALETTE = {
    "bg": "#FAFAF8",
    "grid": "#ECE9E3",
    "dark": "#364152",
    "obstacle": "#BFC5CD",
    "slot_edge": "#D28E2A",
    "slot_fill": "#FFF7E8",
    "ego_edge": "#27B9C6",
    "ego_fill": "#E6FAFC",
    "overlap_fill": "#E97A7A",
    "overlap_edge": "#CC6666",
    "arrow": "#95A2B3",
}

FONT_SIZE = 18.0
REFERENCE_SCENES = {
    "bay": (
        Path(ROOT_DIR)
        / "log"
        / "analysis"
        / "dual_inference_gallery_success_20260404_161038"
        / "normal_bay_11850"
        / "dual_summary.json"
    ),
    "parallel": (
        Path(ROOT_DIR)
        / "log"
        / "analysis"
        / "dual_inference_gallery_success_20260404_161038"
        / "normal_parallel_7"
        / "dual_summary.json"
    ),
}


def configure_plot():
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = FONT_SIZE
    matplotlib.rcParams["font.serif"] = [
        "AR PL UMing CN",
        "Noto Serif CJK JP",
        "Noto Sans CJK JP",
        "DejaVu Serif",
    ]
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def ring_coords(shape):
    coords = np.asarray(shape.coords)
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def transform_polygon(local_coords, center_xy=(0.0, 0.0), yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local_coords @ rot.T + np.asarray(center_xy, dtype=np.float64)


def draw_poly(ax, coords, facecolor, edgecolor, linewidth=2.0, linestyle="-", alpha=1.0, zorder=2):
    ax.add_patch(
        PolygonPatch(
            coords,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )
    )


def load_reference_scene(scene_path: Path):
    with open(scene_path, "r") as f:
        return json.load(f)


def solve_translation_for_iou(length, target_iou):
    return length * (1.0 - target_iou) / (1.0 + target_iou)


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def state_box_from_dict(state_dict):
    local_vehicle = ring_coords(VehicleBox).copy()
    return transform_polygon(
        local_vehicle,
        center_xy=(state_dict["x"], state_dict["y"]),
        yaw=state_dict["heading"],
    )


def densify_reverse_trajectory(trajectory_states, subdivisions=10):
    reversed_states = list(reversed(trajectory_states))
    if len(reversed_states) <= 1:
        return reversed_states

    dense_states = []
    for idx in range(len(reversed_states) - 1):
        state_a = reversed_states[idx]
        state_b = reversed_states[idx + 1]
        heading_delta = wrap_angle(state_b["heading"] - state_a["heading"])
        for sub_idx in range(subdivisions):
            t = sub_idx / float(subdivisions)
            dense_states.append(
                {
                    "x": state_a["x"] + (state_b["x"] - state_a["x"]) * t,
                    "y": state_a["y"] + (state_b["y"] - state_a["y"]) * t,
                    "heading": state_a["heading"] + heading_delta * t,
                    "speed": state_a.get("speed", 0.0) + (state_b.get("speed", 0.0) - state_a.get("speed", 0.0)) * t,
                    "steering": state_a.get("steering", 0.0) + (state_b.get("steering", 0.0) - state_a.get("steering", 0.0)) * t,
                }
            )
    dense_states.append(reversed_states[-1])
    return dense_states


def select_actual_scene_stage_infos(reference_payload, specs):
    dense_states = densify_reverse_trajectory(reference_payload["trajectory_states"], subdivisions=10)
    slot_coords = np.asarray(reference_payload["map"]["dest_box"], dtype=np.float64)

    stage_infos = []
    min_search_idx = 0
    for spec in specs:
        selected_idx = None
        selected_iou = None
        selected_box = None
        for idx in range(min_search_idx, len(dense_states)):
            ego_coords = state_box_from_dict(dense_states[idx])
            current_iou = calc_iou(ego_coords, slot_coords)
            if current_iou < spec["threshold"]:
                selected_idx = idx
                selected_iou = current_iou
                selected_box = ego_coords
                break
        if selected_idx is None:
            selected_idx = len(dense_states) - 1
            selected_box = state_box_from_dict(dense_states[selected_idx])
            selected_iou = calc_iou(selected_box, slot_coords)
        stage_infos.append(
            {
                "state": dense_states[selected_idx],
                "ego_coords": selected_box,
                "iou": selected_iou,
                "path_points": np.array([[state["x"], state["y"]] for state in dense_states[: selected_idx + 1]], dtype=np.float64),
            }
        )
        min_search_idx = selected_idx + 1
    return stage_infos


def panel_specifications():
    return [
        {
            "title": "阶段 1",
            "target_iou": max(UNPARK_IOU_STAGE_THRESHOLDS[0] - 0.02, 0.0),
            "threshold": UNPARK_IOU_STAGE_THRESHOLDS[0],
            "detail": f"阈值 < {UNPARK_IOU_STAGE_THRESHOLDS[0]:.2f}",
        },
        {
            "title": "阶段 2",
            "target_iou": max(UNPARK_IOU_STAGE_THRESHOLDS[1] - 0.02, 0.0),
            "threshold": UNPARK_IOU_STAGE_THRESHOLDS[1],
            "detail": f"阈值 < {UNPARK_IOU_STAGE_THRESHOLDS[1]:.2f}",
        },
        {
            "title": "阶段 3",
            "target_iou": max(UNPARK_IOU_STAGE_THRESHOLDS[2] - 0.04, 0.0),
            "threshold": UNPARK_IOU_STAGE_THRESHOLDS[2],
            "detail": f"阈值 < {UNPARK_IOU_STAGE_THRESHOLDS[2]:.2f}",
        },
        {
            "title": "成功脱离",
            "target_iou": max(UNPARK_SUCCESS_IOU - 0.04, 0.0),
            "threshold": UNPARK_SUCCESS_IOU,
            "detail": f"阈值 < {UNPARK_SUCCESS_IOU:.2f}",
        },
    ]


def edge_heading(coords):
    pts = np.asarray(coords, dtype=np.float64)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    vec = pts[1] - pts[0]
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vec / norm


def polygon_center(coords):
    pts = np.asarray(coords, dtype=np.float64)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts.mean(axis=0)


def translate_polygon(coords, offset):
    pts = np.asarray(coords, dtype=np.float64)
    return pts + np.asarray(offset, dtype=np.float64)


def render_schematic_panel(ax, local_vehicle, spec):
    slot_center = np.array([0.0, 0.0], dtype=np.float64)
    slot_coords = transform_polygon(local_vehicle, center_xy=slot_center, yaw=0.0)
    vehicle_length = float(local_vehicle[:, 0].max() - local_vehicle[:, 0].min())
    dx = solve_translation_for_iou(vehicle_length, spec["target_iou"])
    vehicle_center = np.array([dx, 0.0], dtype=np.float64)
    ego_coords = transform_polygon(local_vehicle, center_xy=vehicle_center, yaw=0.0)

    slot_polygon = Polygon(slot_coords)
    ego_polygon = Polygon(ego_coords)
    overlap_polygon = slot_polygon.intersection(ego_polygon)
    measured_iou = calc_iou(ego_coords, slot_coords)

    ax.set_facecolor(PALETTE["bg"])
    ax.grid(True, linestyle="--", linewidth=0.55, color=PALETTE["grid"], alpha=0.45)

    draw_poly(
        ax,
        slot_coords,
        facecolor=PALETTE["slot_fill"],
        edgecolor=PALETTE["slot_edge"],
        linewidth=2.0,
        linestyle=(0, (4.5, 2.2)),
        alpha=0.95,
        zorder=2,
    )
    draw_poly(
        ax,
        ego_coords,
        facecolor=PALETTE["ego_fill"],
        edgecolor=PALETTE["ego_edge"],
        linewidth=2.2,
        alpha=0.82,
        zorder=3,
    )

    if not overlap_polygon.is_empty:
        overlap_coords = np.asarray(overlap_polygon.exterior.coords)
        draw_poly(
            ax,
            overlap_coords,
            facecolor=PALETTE["overlap_fill"],
            edgecolor=PALETTE["overlap_edge"],
            linewidth=1.4,
            alpha=0.42,
            zorder=4,
        )

    ax.annotate(
        "",
        xy=(vehicle_center[0] + vehicle_length * 0.62, 0.0),
        xytext=(slot_center[0] + vehicle_length * 0.18, 0.0),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=PALETTE["arrow"]),
        zorder=5,
    )

    ax.text(
        0.0,
        -2.25,
        f"IoU = {measured_iou:.2f}\n{spec['detail']}",
        ha="center",
        va="top",
        fontsize=FONT_SIZE - 1.0,
        color=PALETTE["dark"],
    )
    ax.set_title(spec["title"], fontsize=FONT_SIZE + 1.0, color=PALETTE["dark"], pad=10.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.4, 5.8)
    ax.set_ylim(-2.8, 2.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def visible_scene_obstacles(scene_map, focus_poly):
    visible = []
    for obstacle in scene_map["obstacles"]:
        obstacle_poly = Polygon(obstacle)
        if obstacle_poly.intersects(focus_poly):
            visible.append(np.asarray(obstacle, dtype=np.float64))
    return visible


def scene_bounds(slot_coords, ego_coords, obstacles, path_points=None):
    all_pts = [np.asarray(slot_coords, dtype=np.float64), np.asarray(ego_coords, dtype=np.float64)]
    if path_points is not None and len(path_points) > 0:
        all_pts.append(np.asarray(path_points, dtype=np.float64))
    all_pts.extend(np.asarray(obstacle, dtype=np.float64) for obstacle in obstacles)
    stacked = np.vstack(all_pts)
    pad_x = 1.6
    pad_y = 1.6
    return {
        "xmin": float(stacked[:, 0].min() - pad_x),
        "xmax": float(stacked[:, 0].max() + pad_x),
        "ymin": float(stacked[:, 1].min() - pad_y),
        "ymax": float(stacked[:, 1].max() + pad_y),
    }


def render_actual_scene_panel(
    ax,
    scene_map,
    spec,
    stage_info,
    show_stage_title=False,
    show_iou_text=False,
    show_path_prefix=True,
    show_direction_arrow=True,
):
    slot_coords = np.asarray(scene_map["dest_box"], dtype=np.float64)
    ego_coords = np.asarray(stage_info["ego_coords"], dtype=np.float64)
    path_points = np.asarray(stage_info["path_points"], dtype=np.float64)

    slot_polygon = Polygon(slot_coords)
    ego_polygon = Polygon(ego_coords)
    overlap_polygon = slot_polygon.intersection(ego_polygon)

    focus_poly = slot_polygon.union(ego_polygon).buffer(6.0, cap_style=2, join_style=2)
    obstacles = visible_scene_obstacles(scene_map, focus_poly)
    bounds = scene_bounds(slot_coords, ego_coords, obstacles, path_points=path_points)

    ax.set_facecolor(PALETTE["bg"])
    ax.grid(True, linestyle="--", linewidth=0.55, color=PALETTE["grid"], alpha=0.45)

    for obstacle in obstacles:
        draw_poly(
            ax,
            obstacle,
            facecolor=PALETTE["obstacle"],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.96,
            zorder=1,
        )

    if show_path_prefix and len(path_points) >= 2:
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            color=PALETTE["ego_edge"],
            linewidth=2.0,
            alpha=0.92,
            zorder=2.6,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    draw_poly(
        ax,
        slot_coords,
        facecolor=PALETTE["slot_fill"],
        edgecolor=PALETTE["slot_edge"],
        linewidth=2.0,
        linestyle=(0, (4.5, 2.2)),
        alpha=0.90,
        zorder=3,
    )
    draw_poly(
        ax,
        ego_coords,
        facecolor=PALETTE["ego_fill"],
        edgecolor=PALETTE["ego_edge"],
        linewidth=2.0,
        alpha=0.78,
        zorder=4,
    )
    if not overlap_polygon.is_empty:
        overlap_coords = np.asarray(overlap_polygon.exterior.coords)
        draw_poly(
            ax,
            overlap_coords,
            facecolor=PALETTE["overlap_fill"],
            edgecolor=PALETTE["overlap_edge"],
            linewidth=1.2,
            alpha=0.38,
            zorder=5,
        )

    if show_direction_arrow and len(path_points) >= 2:
        ax.annotate(
            "",
            xy=path_points[-1],
            xytext=path_points[-2],
            arrowprops=dict(arrowstyle="->", lw=1.8, color=PALETTE["arrow"]),
            zorder=6,
        )

    if show_stage_title:
        ax.set_title(spec["title"], fontsize=FONT_SIZE + 1.0, color=PALETTE["dark"], pad=8.0)
    if show_iou_text:
        ax.text(
            0.04,
            0.06,
            f"当前值 = {stage_info['iou']:.2f}\n{spec['detail']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=FONT_SIZE - 1.5,
            color=PALETTE["dark"],
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFFFFFCC", edgecolor="none"),
            zorder=7,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bounds["xmin"], bounds["xmax"])
    ax.set_ylim(bounds["ymin"], bounds["ymax"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_legend(fig):
    handles = [
        plt.Line2D([0], [0], color=PALETTE["slot_edge"], lw=2.0, linestyle=(0, (4.5, 2.2))),
        plt.Line2D([0], [0], color=PALETTE["ego_edge"], lw=2.2),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["overlap_fill"], edgecolor=PALETTE["overlap_edge"], alpha=0.42),
    ]
    labels = ["原车位边界", "当前车体", "交集区域"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE - 1.0,
        bbox_to_anchor=(0.5, 0.03),
    )


def render_actual_only(output_dir: Path, stem: str, dpi: int, reference_scene_key: str):
    configure_plot()
    reference_payload = load_reference_scene(REFERENCE_SCENES[reference_scene_key])
    scene_map = reference_payload["map"]
    specs = panel_specifications()
    stage_infos = select_actual_scene_stage_infos(reference_payload, specs)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.4), dpi=dpi)
    for ax, spec, stage_info in zip(axes.flat, specs, stage_infos):
        render_actual_scene_panel(
            ax,
            scene_map,
            spec,
            stage_info,
            show_stage_title=True,
            show_iou_text=False,
            show_path_prefix=False,
            show_direction_arrow=False,
        )

    add_legend(fig)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.10, wspace=0.04, hspace=0.10)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

    print(output_dir / f"{stem}.png")
    print(output_dir / f"{stem}.pdf")
    print(output_dir / f"{stem}.svg")


def render(output_dir: Path, stem: str, dpi: int, reference_scene_key: str):
    configure_plot()
    local_vehicle = ring_coords(VehicleBox).copy()
    reference_payload = load_reference_scene(REFERENCE_SCENES[reference_scene_key])
    scene_map = reference_payload["map"]

    fig, axes = plt.subplots(2, 4, figsize=(18.2, 8.3), dpi=dpi)
    specs = panel_specifications()
    stage_infos = select_actual_scene_stage_infos(reference_payload, specs)
    for col, spec in enumerate(specs):
        render_schematic_panel(axes[0, col], local_vehicle, spec)
        render_actual_scene_panel(axes[1, col], scene_map, spec, stage_infos[col])

    fig.suptitle("泊出任务中的交并比阶段示意", fontsize=FONT_SIZE + 3.0, color=PALETTE["dark"], y=0.985)
    fig.text(
        0.5,
        0.915,
        "交并比 = 交集面积 / 并集面积",
        ha="center",
        va="center",
        fontsize=FONT_SIZE - 0.5,
        color=PALETTE["dark"],
    )
    fig.text(0.015, 0.64, "抽象示意", rotation=90, ha="center", va="center", fontsize=FONT_SIZE + 1.0, color=PALETTE["dark"])
    fig.text(0.015, 0.31, "实际场景", rotation=90, ha="center", va="center", fontsize=FONT_SIZE + 1.0, color=PALETTE["dark"])
    add_legend(fig)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.15, wspace=0.09, hspace=0.18)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

    print(output_dir / f"{stem}.png")
    print(output_dir / f"{stem}.pdf")
    print(output_dir / f"{stem}.svg")


def main():
    parser = argparse.ArgumentParser(description="导出泊出任务 IoU 阶段示意图。")
    parser.add_argument("--output-dir", type=str, default=str(Path(ROOT_DIR) / "log" / "paper_support"))
    parser.add_argument("--stem", type=str, default="unpark_iou_schematic_cn")
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--reference-scene", type=str, choices=sorted(REFERENCE_SCENES.keys()), default="parallel")
    parser.add_argument("--actual-only", action="store_true")
    args = parser.parse_args()

    if args.actual_only:
        render_actual_only(
            Path(args.output_dir),
            args.stem,
            max(int(args.dpi), 100),
            args.reference_scene,
        )
    else:
        render(
            Path(args.output_dir),
            args.stem,
            max(int(args.dpi), 100),
            args.reference_scene,
        )


if __name__ == "__main__":
    main()
