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


STYLE_PRESETS = {
    "minimal": {
        "bg": "#FBFBF8",
        "obstacle": "#50535A",
        "obstacle_edge": "white",
        "obstacle_alpha": 0.98,
        "slot_fill": "#3AB7B4",
        "slot_edge": "#0E7C86",
        "slot_alpha": 0.95,
        "goal_fill": "#F2C078",
        "goal_edge": "#C8640C",
        "goal_alpha": 0.95,
        "valid": "#2FBF71",
        "invalid": "#E03131",
        "show_axes": False,
        "grid_color": "#E6E6E6",
        "grid_alpha": 0.7,
        "spine_color": "#D0D0D0",
    },
    "reference_grid": {
        "bg": "#FFFFFF",
        "obstacle": "#B8B8B8",
        "obstacle_edge": "#B0B0B0",
        "obstacle_alpha": 0.92,
        "slot_fill": "#FFFFFF",
        "slot_edge": "#1DA7B8",
        "slot_alpha": 1.0,
        "goal_fill": "#FFFFFF",
        "goal_edge": "#F28E2B",
        "goal_alpha": 1.0,
        "valid": "#12B3B6",
        "invalid": "#E15759",
        "show_axes": True,
        "grid_color": "#DADADA",
        "grid_alpha": 0.75,
        "spine_color": "#BFBFBF",
    },
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


def draw_polygon(ax, coords, facecolor, edgecolor, linewidth=1.5, alpha=0.95, linestyle="-", zorder=3):
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


def family_name(ctypes):
    return "".join("C" if char in {"L", "R"} else char for char in ctypes)


def path_label(path):
    return f"{family_name(path.ctypes)} ({rs_curve.get_label(path)})"


def load_case(trace_path):
    trace = json.load(open(trace_path, "r", encoding="utf-8"))
    connection_step = trace.get("connection_step")
    connection_index = trace.get("connection_index")
    if connection_step is None or connection_index is None:
        raise ValueError("This trace does not contain a successful RS connection.")

    current_state = state_from_dict(trace["trajectory_states"][connection_step])
    goal_state = state_from_dict(trace["reverse_states"][connection_index])
    return trace, current_state, goal_state


def collect_distinct_family_paths(trace, current_state, goal_state):
    env = CarParkingWrapper(CarParking(fps=0, verbose=False, render_mode="rgb_array"))
    env.unwrapped.map = build_map(trace["map"])
    env.unwrapped.vehicle.reset(current_state)
    env.unwrapped.matrix = env.unwrapped.coord_transform_matrix()

    radius = math.tan(VALID_STEER[-1]) / WHEEL_BASE
    all_paths = rs_curve.calc_all_paths(
        current_state.loc.x,
        current_state.loc.y,
        current_state.heading,
        goal_state.loc.x,
        goal_state.loc.y,
        goal_state.heading,
        radius,
        0.1,
    )
    all_paths = sorted(all_paths, key=lambda path: path.L)

    selected = []
    seen_families = set()
    for path in all_paths:
        family = family_name(path.ctypes)
        if family in seen_families:
            continue
        traj = [[path.x[idx], path.y[idx], path.yaw[idx]] for idx in range(len(path.x))]
        valid = env.unwrapped.is_traj_valid(traj)
        selected.append(
            {
                "family": family,
                "rs_label": rs_curve.get_label(path),
                "length": path.L,
                "valid": valid,
                "path": path,
            }
        )
        seen_families.add(family)

    env.close()
    return selected


def select_paths(distinct_paths, topk, policy):
    if policy == "topk_distinct":
        return distinct_paths[:topk]

    if policy == "csc_valid_then_invalid":
        chosen = []
        csc_item = next((item for item in distinct_paths if item["family"] == "CSC" and item["valid"]), None)
        if csc_item is None:
            raise ValueError("No valid CSC path found in this trace.")
        chosen.append(csc_item)

        for item in distinct_paths:
            if item["family"] == "CSC":
                continue
            if item["valid"]:
                continue
            chosen.append(item)
            if len(chosen) >= topk:
                break

        if len(chosen) < topk:
            raise ValueError("Not enough invalid non-CSC path families for the requested topk.")
        return chosen

    raise ValueError(f"Unknown policy: {policy}")


def render(trace_path, output_dir, stem, topk, policy, show_obstacles, style):
    configure_plot()
    trace, current_state, goal_state = load_case(trace_path)
    distinct_paths = collect_distinct_family_paths(trace, current_state, goal_state)
    top_paths = select_paths(distinct_paths, topk, policy)
    map_dict = trace["map"]
    palette = STYLE_PRESETS[style]

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 7.0), dpi=330)
    ax.set_facecolor(palette["bg"])

    if show_obstacles:
        for obstacle in map_dict["obstacles"]:
            draw_polygon(
                ax,
                obstacle,
                palette["obstacle"],
                palette["obstacle_edge"],
                linewidth=0.8,
                alpha=palette["obstacle_alpha"],
                zorder=2,
            )

    draw_polygon(
        ax,
        map_dict["dest_box"],
        palette["slot_fill"],
        palette["slot_edge"],
        linewidth=1.9,
        alpha=palette["slot_alpha"],
        zorder=3,
    )
    draw_polygon(
        ax,
        ring_coords(current_state.create_box()),
        palette["goal_fill"],
        palette["goal_edge"],
        linewidth=1.9,
        alpha=palette["goal_alpha"],
        zorder=4,
    )

    for item in top_paths:
        color = palette["valid"] if item["valid"] else palette["invalid"]
        linestyle = "-" if item["valid"] else "--"
        alpha = 0.96 if item["valid"] else 0.88
        path = item["path"]
        ax.plot(path.x, path.y, color=color, linewidth=3.0, linestyle=linestyle, alpha=alpha, zorder=5)

    bounds = map_dict["bounds"]
    ax.set_xlim(bounds["xmin"] - 1.0, bounds["xmax"] + 1.0)
    ax.set_ylim(bounds["ymin"] - 1.0, bounds["ymax"] + 1.0)
    ax.set_aspect("equal", adjustable="box")
    if palette["show_axes"]:
        ax.grid(True, color=palette["grid_color"], linewidth=0.7, alpha=palette["grid_alpha"])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(palette["spine_color"])
            spine.set_linewidth(0.8)
        ax.tick_params(labelsize=10, colors="#4A4A4A")
    else:
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
    parser.add_argument("--stem", type=str, default="rs_curve_type_topk_extrem_seed1")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--policy", type=str, default="topk_distinct", choices=["topk_distinct", "csc_valid_then_invalid"])
    parser.add_argument("--show-obstacles", action="store_true")
    parser.add_argument("--style", type=str, default="minimal", choices=["minimal", "reference_grid"])
    args = parser.parse_args()

    png_path, pdf_path = render(
        args.trace,
        args.output_dir,
        args.stem,
        args.topk,
        args.policy,
        args.show_obstacles,
        args.style,
    )
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
