# -*- coding: utf-8 -*-
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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as PolygonPatch
from shapely.geometry import LinearRing

from configs import VALID_STEER, VehicleBox, WHEEL_BASE
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.map_base import Area
from env.parking_map_normal import ParkingMapNormal
from env.vehicle import State
import env.reeds_shepp as rs_curve


PALETTE = {
    "bg": "#FFFFFF",
    "obstacle_fill": "#BDBDBD",
    "obstacle_edge": "#B3B3B3",
    "start_edge": "#F28E2B",
    "start_fill": "#FFFFFF",
    "goal_edge": "#16B6C8",
    "goal_fill": "#FFFFFF",
    "start_dot": "#F28E2B",
    "goal_dot": "#16B6C8",
    "valid": "#19B7B3",
    "invalid": "#E45B5B",
}

DISPLAY_CAR_LENGTH_SCALE = 0.85


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


def draw_polygon(ax, coords, facecolor, edgecolor, linewidth=1.5, alpha=1.0, linestyle="-", zorder=3):
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


def rect_polygon(cx, cy, w, h, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[-w / 2.0, -h / 2.0], [w / 2.0, -h / 2.0], [w / 2.0, h / 2.0], [-w / 2.0, h / 2.0]])
    rot = np.array([[c, -s], [s, c]])
    pts = local @ rot.T + np.array([cx, cy])
    return pts.tolist() + [pts[0].tolist()]


def display_vehicle_polygon(state, length_scale=DISPLAY_CAR_LENGTH_SCALE):
    # Display-only shortening while keeping the rear-wheel center fixed.
    local = ring_coords(VehicleBox).copy()
    local[:, 0] *= length_scale
    c, s = math.cos(state.heading), math.sin(state.heading)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([state.loc.x, state.loc.y])


def build_schematic_case():
    # Project states use the rear-wheel center as the pose origin.
    start = State([-7.924, 0.628, -0.468])
    goal = State([2.808, 6.431, 1.748])
    obstacles = [
        rect_polygon(-5.783001054289061, 6.728820923259562, 2.3065829120458382, 1.4415207281368912, 0.5083252525209118),
        rect_polygon(-1.0422263008903178, 8.040766870292086, 3.7697151905306815, 1.4115903515279572, -0.273418223429893),
        # Lift and enlarge the lower block further so the failure contrast is more
        # obvious while keeping only the SCS connector feasible in this schematic.
        rect_polygon(-0.7033604364972002, 1.1102146049300503, 4.5699339288883935, 1.8121185847790009, -0.90550132396849053),
        rect_polygon(-0.6120227662117994, 5.24394451048252, 2.293500309892192, 1.0071333009788549, -0.24115542720504595),
    ]
    bounds = {"xmin": -12.0, "xmax": 10.0, "ymin": -4.0, "ymax": 14.0}
    return {"start": start, "goal": goal, "obstacles": obstacles, "bounds": bounds}


def build_map(case):
    parking_map = ParkingMapNormal("Normal")
    bounds = case["bounds"]
    parking_map.xmin = bounds["xmin"]
    parking_map.xmax = bounds["xmax"]
    parking_map.ymin = bounds["ymin"]
    parking_map.ymax = bounds["ymax"]
    parking_map.start = case["start"]
    parking_map.dest = case["goal"]
    parking_map.start_box = parking_map.start.create_box()
    parking_map.dest_box = parking_map.dest.create_box()
    parking_map.obstacles = [Area(shape=LinearRing(obstacle), subtype="obstacle", color=(150, 150, 150, 255)) for obstacle in case["obstacles"]]
    parking_map.n_obstacle = len(parking_map.obstacles)
    return parking_map


def family_name(ctypes):
    return "".join("C" if token in {"L", "R"} else token for token in ctypes)


def compute_topk_paths(case, topk):
    env = CarParkingWrapper(CarParking(fps=0, verbose=False, render_mode="rgb_array"))
    env.unwrapped.map = build_map(case)
    env.unwrapped.vehicle.reset(case["start"])
    env.unwrapped.matrix = env.unwrapped.coord_transform_matrix()

    radius = math.tan(VALID_STEER[-1]) / WHEEL_BASE
    all_paths = rs_curve.calc_all_paths(
        case["start"].loc.x,
        case["start"].loc.y,
        case["start"].heading,
        case["goal"].loc.x,
        case["goal"].loc.y,
        case["goal"].heading,
        radius,
        0.1,
    )
    all_paths = sorted(all_paths, key=lambda path: path.L)

    selected = []
    seen = set()
    for path in all_paths:
        family = family_name(path.ctypes)
        if family in seen:
            continue
        traj = [[path.x[idx], path.y[idx], path.yaw[idx]] for idx in range(len(path.x))]
        selected.append(
            {
                "family": family,
                "valid": env.unwrapped.is_traj_valid(traj),
                "length": path.L,
                "rs_label": rs_curve.get_label(path),
                "path": path,
            }
        )
        seen.add(family)
        if len(selected) >= topk:
            break

    env.close()
    return selected


def render(output_dir, stem, topk, show_legend=False):
    configure_plot()
    case = build_schematic_case()
    top_paths = compute_topk_paths(case, topk)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=400)
    ax.set_facecolor(PALETTE["bg"])

    for obstacle in case["obstacles"]:
        draw_polygon(
            ax,
            obstacle,
            PALETTE["obstacle_fill"],
            PALETTE["obstacle_edge"],
            linewidth=0.9,
            alpha=0.95,
            zorder=2,
        )

    draw_polygon(
        ax,
        display_vehicle_polygon(case["start"]),
        PALETTE["start_fill"],
        PALETTE["start_edge"],
        linewidth=1.9,
        alpha=1.0,
        zorder=4,
    )
    draw_polygon(
        ax,
        display_vehicle_polygon(case["goal"]),
        PALETTE["goal_fill"],
        PALETTE["goal_edge"],
        linewidth=1.9,
        alpha=1.0,
        zorder=4,
    )

    ax.scatter(case["start"].loc.x, case["start"].loc.y, s=28, color=PALETTE["start_dot"], zorder=6)
    ax.scatter(case["goal"].loc.x, case["goal"].loc.y, s=28, color=PALETTE["goal_dot"], zorder=6)

    for item in top_paths:
        color = PALETTE["valid"] if item["valid"] else PALETTE["invalid"]
        linestyle = "-" if item["valid"] else "--"
        linewidth = 3.0 if item["valid"] else 2.7
        alpha = 0.96 if item["valid"] else 0.92
        ax.plot(item["path"].x, item["path"].y, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=5)

    if show_legend:
        legend_handles = [
            Patch(facecolor=PALETTE["start_fill"], edgecolor=PALETTE["start_edge"], linewidth=1.8, label="Start Pose"),
            Patch(facecolor=PALETTE["goal_fill"], edgecolor=PALETTE["goal_edge"], linewidth=1.8, label="Goal Pose"),
            Line2D([0], [0], color=PALETTE["valid"], lw=3.0, linestyle="-", label="Feasible RS"),
            Line2D([0], [0], color=PALETTE["invalid"], lw=2.7, linestyle="--", label="Infeasible RS"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.96,
            facecolor="white",
            edgecolor="#D0D0D0",
            fontsize=10,
            handlelength=2.2,
            labelspacing=0.6,
        )

    bounds = case["bounds"]
    ax.set_xlim(bounds["xmin"], bounds["xmax"])
    ax.set_ylim(bounds["ymin"], bounds["ymax"])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.2)

    png_path = Path(output_dir) / f"{stem}.png"
    pdf_path = Path(output_dir) / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="src/log/paper_support")
    parser.add_argument("--stem", type=str, default="rs_topk_manual_schematic")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--show-legend", action="store_true")
    args = parser.parse_args()

    png_path, pdf_path = render(args.output_dir, args.stem, args.topk, show_legend=args.show_legend)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
