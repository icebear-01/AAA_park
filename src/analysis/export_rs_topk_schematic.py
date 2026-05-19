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
from shapely.geometry import LinearRing

from configs import VALID_STEER, WHEEL_BASE
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.map_base import Area
from env.parking_map_normal import ParkingMapNormal
from env.vehicle import State
import env.reeds_shepp as rs_curve


PALETTE = {
    "bg": "#FBFBF8",
    "obstacle": "#50535A",
    "dest_fill": "#3AB7B4",
    "dest_edge": "#0E7C86",
    "ego_edge": "#C8640C",
    "anchor": "#144D8A",
    "anchor_text": "#0F2747",
    "valid_rs": "#2FBF71",
    "invalid_rs": "#E03131",
    "selected_rs": "#178F5C",
}


def configure_plot():
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def ring_coords(shape_or_area):
    shape = shape_or_area.shape if hasattr(shape_or_area, "shape") else shape_or_area
    coords = np.asarray(shape.coords)
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def build_map(map_dict):
    parking_map = ParkingMapNormal(map_dict["map_level"])
    bounds = map_dict["bounds"]
    parking_map.xmin = bounds["xmin"]
    parking_map.xmax = bounds["xmax"]
    parking_map.ymin = bounds["ymin"]
    parking_map.ymax = bounds["ymax"]

    start = map_dict["start"]
    dest = map_dict["dest"]
    parking_map.start = State([start["x"], start["y"], start["heading"], start.get("speed", 0.0), start.get("steering", 0.0)])
    parking_map.dest = State([dest["x"], dest["y"], dest["heading"], dest.get("speed", 0.0), dest.get("steering", 0.0)])
    parking_map.start_box = parking_map.start.create_box()
    parking_map.dest_box = parking_map.dest.create_box()
    parking_map.obstacles = [
        Area(shape=LinearRing(obstacle), subtype="obstacle", color=(150, 150, 150, 255))
        for obstacle in map_dict["obstacles"]
    ]
    parking_map.n_obstacle = len(parking_map.obstacles)
    return parking_map


def state_from_dict(state_dict):
    return State(
        [
            state_dict["x"],
            state_dict["y"],
            state_dict["heading"],
            state_dict.get("speed", 0.0),
            state_dict.get("steering", 0.0),
        ]
    )


def candidate_indices(n_states, connect_stride):
    dense_tail = min(4, n_states)
    indices = list(range(n_states - 1, n_states - dense_tail - 1, -1))
    stride_start = n_states - dense_tail - 1
    if stride_start >= 0:
        indices.extend(range(stride_start, -1, -connect_stride))
    if indices[-1] != 0:
        indices.append(0)
    return list(dict.fromkeys(indices))


def make_env(map_dict, current_state):
    raw_env = CarParking(fps=0, verbose=False, render_mode="rgb_array")
    env = CarParkingWrapper(raw_env)
    env.unwrapped.map = build_map(map_dict)
    env.unwrapped.vehicle.reset(current_state)
    env.unwrapped.matrix = env.unwrapped.coord_transform_matrix()
    return env


def draw_box(ax, coords, facecolor, edgecolor, linewidth=1.5, alpha=0.95, linestyle="-", zorder=3):
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


def plot_path(ax, path, color, linewidth, linestyle="-", alpha=1.0, zorder=5):
    ax.plot(path.x, path.y, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder)


def compute_paths(trace, connect_stride=5):
    connection_step = trace.get("connection_step")
    current_state_dict = trace["trajectory_states"][connection_step if connection_step is not None else 0]
    current_state = state_from_dict(current_state_dict)
    env = make_env(trace["map"], current_state)

    radius = math.tan(VALID_STEER[-1]) / WHEEL_BASE
    reverse_states = [state_from_dict(state_dict) for state_dict in trace["reverse_states"]]
    candidates = candidate_indices(len(reverse_states), connect_stride)

    evaluated = []
    for rank, state_index in enumerate(candidates, start=1):
        goal_state = reverse_states[state_index]
        valid_path = env.unwrapped.find_rs_path_to_state(goal_state, start_state=current_state)
        raw_path = rs_curve.calc_optimal_path(
            current_state.loc.x,
            current_state.loc.y,
            current_state.heading,
            goal_state.loc.x,
            goal_state.loc.y,
            goal_state.heading,
            radius,
            0.1,
        )
        evaluated.append(
            {
                "rank": rank,
                "state_index": state_index,
                "goal_state": goal_state,
                "valid": valid_path is not None,
                "path": valid_path if valid_path is not None else raw_path,
            }
        )

    env.close()
    return current_state, reverse_states, evaluated


def render(trace_path, output_dir, stem, connect_stride=5):
    configure_plot()
    trace = json.load(open(trace_path, "r", encoding="utf-8"))
    current_state, reverse_states, evaluated = compute_paths(trace, connect_stride=connect_stride)

    map_dict = trace["map"]
    chosen_index = trace.get("connection_index")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 7.0), dpi=330)
    ax.set_facecolor(PALETTE["bg"])

    for obstacle in map_dict["obstacles"]:
        draw_box(ax, obstacle, PALETTE["obstacle"], "white", linewidth=0.8, alpha=0.98, zorder=2)

    draw_box(ax, map_dict["dest_box"], PALETTE["dest_fill"], PALETTE["dest_edge"], linewidth=1.8, alpha=0.95, zorder=3)
    start_xy = np.array([current_state.loc.x, current_state.loc.y], dtype=np.float64)
    ax.scatter(start_xy[0], start_xy[1], s=140, marker="*", color=PALETTE["ego_edge"], edgecolor="white", linewidth=1.0, zorder=8)
    ax.text(
        start_xy[0] - 0.2,
        start_xy[1] - 0.9,
        "当前状态",
        fontsize=12,
        color=PALETTE["ego_edge"],
        ha="center",
        va="top",
        zorder=9,
    )

    for item in evaluated:
        color = PALETTE["valid_rs"] if item["valid"] else PALETTE["invalid_rs"]
        linestyle = "-" if item["valid"] else "--"
        linewidth = 3.2 if item["state_index"] == chosen_index else 1.9
        alpha = 0.95 if item["valid"] else 0.85
        if item["path"] is not None:
            plot_path(ax, item["path"], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=5)
        center = np.array([item["goal_state"].loc.x, item["goal_state"].loc.y], dtype=np.float64)
        marker_size = 46 if item["state_index"] == chosen_index else 30
        edge_width = 1.0 if item["state_index"] == chosen_index else 0.7
        ax.scatter(center[0], center[1], s=marker_size, color=PALETTE["anchor"], edgecolor="white", linewidth=edge_width, zorder=7)
        ax.text(
            center[0] + 0.18,
            center[1] + 0.18,
            str(item["rank"]),
            fontsize=12,
            fontweight="bold",
            color=PALETTE["anchor_text"],
            zorder=8,
        )

    dest_center = np.mean(np.asarray(map_dict["dest_box"]), axis=0)
    ax.annotate("终点车位环境", xy=(dest_center[0], dest_center[1]), xytext=(dest_center[0] + 1.4, dest_center[1] + 2.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["dest_edge"], lw=1.2),
                fontsize=12, color=PALETTE["dest_edge"])

    valid_count = sum(1 for item in evaluated if item["valid"])
    invalid_count = len(evaluated) - valid_count
    title = (
        f"真实场景 RS Top-K 连接示意\n"
        f"{Path(trace_path).parent.name} | valid={valid_count}, invalid={invalid_count}, chosen idx={chosen_index}"
    )
    ax.set_title(title, fontsize=16)
    ax.text(
        0.01,
        0.01,
        "绿色实线: 未判碰且可连接   红色虚线: RS 候选存在但碰撞/越界不可用",
        transform=ax.transAxes,
        fontsize=11,
        color="#243447",
        ha="left",
        va="bottom",
    )

    bounds = map_dict["bounds"]
    ax.set_xlim(bounds["xmin"] - 1.0, bounds["xmax"] + 1.0)
    ax.set_ylim(bounds["ymin"] - 1.0, bounds["ymax"] + 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axis("off")
    fig.tight_layout()

    png_path = Path(output_dir) / f"{stem}.png"
    pdf_path = Path(output_dir) / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        type=str,
        default="src/log/analysis/vector_case_plots_20260330_090845/extrem_seed1/dual_trace.json",
    )
    parser.add_argument("--output-dir", type=str, default="src/log/paper_support")
    parser.add_argument("--stem", type=str, default="rs_topk_actual_scene")
    parser.add_argument("--connect-stride", type=int, default=5)
    args = parser.parse_args()

    png_path, pdf_path = render(args.trace, args.output_dir, args.stem, connect_stride=args.connect_stride)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
