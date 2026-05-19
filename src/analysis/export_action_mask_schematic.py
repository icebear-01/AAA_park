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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon as PolygonPatch
from shapely.geometry import LinearRing

from configs import NUM_STEP, VehicleBox, discrete_actions
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.map_base import Area
from env.parking_map_normal import ParkingMapNormal
from env.vehicle import State


PALETTE = {
    "bg": "#FAFAF8",
    "grid": "#E6E8EC",
    "dark": "#2F3B4A",
    "obstacle_fill": "#BFC5CD",
    "obstacle_edge": "#B2B8C0",
    "ego_edge": "#C98A2E",
    "ego_fill": "#FFFFFF",
    "ego_dot": "#C98A2E",
    "unsafe": "#CC6677",
    "safe": "#3A7FB7",
}

MASK_CMAP = LinearSegmentedColormap.from_list(
    "action_mask_paper",
    ["#CC6677", "#F8F6F2", "#3A7FB7"],
)

DISPLAY_OBSTACLE_SCALE = 1.14
USE_COLLISION_CAR_FOOTPRINT = False
DISPLAY_CAR_LENGTH_SCALE = 0.72
DISPLAY_CAR_WIDTH_SCALE = 0.78
UNIFORM_FONT_SIZE = 20.0


def configure_plot():
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = UNIFORM_FONT_SIZE
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


def ring_coords(shape_or_area):
    shape = shape_or_area.shape if hasattr(shape_or_area, "shape") else shape_or_area
    coords = np.asarray(shape.coords)
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def rect_polygon(cx, cy, w, h, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array(
        [
            [-w / 2.0, -h / 2.0],
            [w / 2.0, -h / 2.0],
            [w / 2.0, h / 2.0],
            [-w / 2.0, h / 2.0],
        ]
    )
    rot = np.array([[c, -s], [s, c]])
    pts = local @ rot.T + np.array([cx, cy])
    return pts.tolist() + [pts[0].tolist()]


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


def scale_polygon(coords, scale):
    points = np.asarray(coords, dtype=np.float64)
    base = points[:-1] if len(points) > 1 and np.allclose(points[0], points[-1]) else points
    center = base.mean(axis=0)
    return (points - center) * scale + center


def display_vehicle_polygon(state):
    local = ring_coords(VehicleBox).copy()
    if not USE_COLLISION_CAR_FOOTPRINT:
        local[:, 0] *= DISPLAY_CAR_LENGTH_SCALE
        local[:, 1] *= DISPLAY_CAR_WIDTH_SCALE
    c, s = math.cos(state.heading), math.sin(state.heading)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([state.loc.x, state.loc.y])


def build_mask_case():
    start = State([0.0, 0.0, 0.0])
    dest = State([8.0, 3.0, 0.0])
    obstacles = [
        rect_polygon(5.3, 2.1, 2.7, 1.1, 0.35),
        rect_polygon(6.8, -0.4, 2.8, 1.1, -0.08),
        rect_polygon(0.8, 2.4, 2.0, 1.0, -0.50),
        rect_polygon(-2.5, -2.5, 2.5, 1.6, 0.90),
    ]
    bounds = {"xmin": -6.5, "xmax": 8.6, "ymin": -5.2, "ymax": 5.8}
    return {"start": start, "dest": dest, "obstacles": obstacles, "bounds": bounds}


def build_map(case):
    parking_map = ParkingMapNormal("Normal")
    bounds = case["bounds"]
    parking_map.xmin = bounds["xmin"]
    parking_map.xmax = bounds["xmax"]
    parking_map.ymin = bounds["ymin"]
    parking_map.ymax = bounds["ymax"]
    parking_map.start = case["start"]
    parking_map.dest = case["dest"]
    parking_map.start_box = parking_map.start.create_box()
    parking_map.dest_box = parking_map.dest.create_box()
    parking_map.obstacles = [
        Area(shape=LinearRing(obstacle), subtype="obstacle", color=(150, 150, 150, 255))
        for obstacle in case["obstacles"]
    ]
    parking_map.n_obstacle = len(parking_map.obstacles)
    return parking_map


def rollout_action(kinematic_model, start_state, action, horizon, step_time=NUM_STEP):
    states = [start_state]
    state = start_state
    for _ in range(horizon):
        state = kinematic_model.step(state, action, step_time=step_time)
        states.append(state)
    return states


def compute_mask_and_rollouts(case, sample_stride, display_horizon=10):
    env = CarParkingWrapper(CarParking(fps=0, verbose=False, render_mode="rgb_array"))
    env.unwrapped.map = build_map(case)
    env.unwrapped.vehicle.reset(case["start"])
    env.unwrapped.matrix = env.unwrapped.coord_transform_matrix()

    model = env.unwrapped.vehicle.kinetic_model
    display_step_time = NUM_STEP

    exact_safe_steps = np.zeros(len(discrete_actions), dtype=np.int32)
    for idx, action in enumerate(discrete_actions):
        env.unwrapped.vehicle.reset(case["start"])
        safe = 0
        for _ in range(display_horizon):
            prev_info = env.unwrapped.vehicle.step(action, step_time=display_step_time)
            # For the schematic we only keep obstacle collision as the validity
            # criterion and ignore map-boundary truncation.
            if env.unwrapped._detect_collision():
                env.unwrapped.vehicle.retreat(prev_info)
                break
            safe += 1
        exact_safe_steps[idx] = safe
    action_mask = exact_safe_steps.astype(np.float64) / float(display_horizon)

    sampled_indices = list(range(0, 21, sample_stride)) + list(range(21, 42, sample_stride))
    rollouts = []
    for idx in sampled_indices:
        action = discrete_actions[idx]
        states = rollout_action(model, case["start"], action, display_horizon, step_time=display_step_time)
        points = np.array([[state.loc.x, state.loc.y] for state in states], dtype=np.float64)
        safe_steps = int(exact_safe_steps[idx])
        rollouts.append(
            {
                "index": idx,
                "points": points,
                "mask_value": float(action_mask[idx]),
                "safe_steps": safe_steps,
            }
        )

    env.close()
    return action_mask, rollouts, display_horizon


def style_axes(ax):
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(True, linestyle="--", linewidth=0.55, color=PALETTE["grid"], alpha=0.45)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("left", "bottom"):
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_color(PALETTE["grid"])
        else:
            spine.set_visible(False)


def render(output_dir, stem, sample_stride, dpi, show_scene_axes=False):
    configure_plot()
    case = build_mask_case()
    action_mask, rollouts, display_horizon = compute_mask_and_rollouts(case, sample_stride=sample_stride)

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(11.8, 4.9), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.16)
    ax_scene = fig.add_subplot(gs[0, 0])
    ax_mask = fig.add_subplot(gs[0, 1])

    style_axes(ax_scene)
    for obstacle in case["obstacles"]:
        draw_polygon(
            ax_scene,
            scale_polygon(obstacle, DISPLAY_OBSTACLE_SCALE),
            PALETTE["obstacle_fill"],
            PALETTE["obstacle_edge"],
            linewidth=1.0,
            alpha=0.95,
            zorder=2,
        )

    for rollout in rollouts:
        points = rollout["points"]
        safe_steps = rollout["safe_steps"]
        ax_scene.plot(
            points[:, 0],
            points[:, 1],
            color=PALETTE["unsafe"],
            linewidth=1.45,
            alpha=0.70,
            linestyle=(0, (4.0, 2.0)),
            zorder=3,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        if safe_steps > 0:
            safe_points = points[: safe_steps + 1]
            ax_scene.plot(
                safe_points[:, 0],
                safe_points[:, 1],
                color=PALETTE["safe"],
                linewidth=2.05,
                alpha=0.98,
                zorder=4,
                solid_capstyle="round",
                solid_joinstyle="round",
            )

    draw_polygon(
        ax_scene,
        display_vehicle_polygon(case["start"]),
        PALETTE["ego_fill"],
        PALETTE["ego_edge"],
        linewidth=2.0,
        alpha=1.0,
        zorder=1,
    )
    ax_scene.scatter(case["start"].loc.x, case["start"].loc.y, s=58, color=PALETTE["ego_dot"], zorder=6)

    bounds = case["bounds"]
    ax_scene.set_xlim(bounds["xmin"], bounds["xmax"])
    ax_scene.set_ylim(bounds["ymin"], bounds["ymax"])
    ax_scene.set_aspect("equal", adjustable="box")
    if show_scene_axes:
        x_ticks = np.arange(math.floor(bounds["xmin"] / 2.0) * 2.0, math.ceil(bounds["xmax"] / 2.0) * 2.0 + 0.1, 2.0)
        y_ticks = np.arange(math.floor(bounds["ymin"] / 2.0) * 2.0, math.ceil(bounds["ymax"] / 2.0) * 2.0 + 0.1, 2.0)
        ax_scene.set_xticks(x_ticks)
        ax_scene.set_yticks(y_ticks)
        ax_scene.tick_params(labelsize=UNIFORM_FONT_SIZE, colors=PALETTE["dark"], length=0)
        ax_scene.set_xlabel("x / m", fontsize=UNIFORM_FONT_SIZE, color=PALETTE["dark"], labelpad=7.0)
        ax_scene.set_ylabel("y / m", fontsize=UNIFORM_FONT_SIZE, color=PALETTE["dark"], labelpad=7.0)
    else:
        ax_scene.set_xticks([])
        ax_scene.set_yticks([])
        ax_scene.set_xlabel("")
        ax_scene.set_ylabel("")
    ax_scene.set_title("局部动作展开", fontsize=UNIFORM_FONT_SIZE, color=PALETTE["dark"], pad=8.0)

    heat = np.vstack([action_mask[:21], action_mask[21:]])
    im = ax_mask.imshow(heat, cmap=MASK_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax_mask.set_title("动作掩码", fontsize=UNIFORM_FONT_SIZE, color=PALETTE["dark"], pad=8.0)
    ax_mask.set_xlabel("转向离散动作", fontsize=UNIFORM_FONT_SIZE, color=PALETTE["dark"], labelpad=8.0)
    ax_mask.set_xticks([0, 10, 20])
    ax_mask.set_xticklabels(["左转", "直行", "右转"], fontsize=UNIFORM_FONT_SIZE, color=PALETTE["dark"])
    ax_mask.set_yticks([0, 1])
    ax_mask.set_yticklabels(["前进", "倒车"], fontsize=UNIFORM_FONT_SIZE, color=PALETTE["dark"])
    ax_mask.tick_params(labelsize=UNIFORM_FONT_SIZE, length=0)

    for row in range(3):
        ax_mask.axhline(row - 0.5, color="white", linewidth=1.0, alpha=0.85)
    for col in range(22):
        ax_mask.axvline(col - 0.5, color="white", linewidth=0.8, alpha=0.85)
    for rollout in rollouts:
        row = 0 if rollout["index"] < 21 else 1
        col = rollout["index"] % 21
        ax_mask.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=PALETTE["dark"],
                linewidth=0.75,
                alpha=0.85,
            )
        )

    for spine_name in ("left", "right", "top", "bottom"):
        ax_mask.spines[spine_name].set_visible(False)

    cbar = fig.colorbar(im, ax=ax_mask, fraction=0.050, pad=0.03)
    cbar.set_label("归一化安全步长", fontsize=UNIFORM_FONT_SIZE, color=PALETTE["dark"])
    cbar.ax.tick_params(labelsize=UNIFORM_FONT_SIZE, colors=PALETTE["dark"])
    fig.subplots_adjust(bottom=0.16, top=0.90, right=0.94)

    fig.savefig(Path(output_dir) / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(Path(output_dir) / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(Path(output_dir) / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

    print(Path(output_dir) / f"{stem}.png")
    print(Path(output_dir) / f"{stem}.pdf")
    print(Path(output_dir) / f"{stem}.svg")


def main():
    parser = argparse.ArgumentParser(description="导出动作掩码的论文示意图。")
    parser.add_argument("--output-dir", type=str, default=str(Path(ROOT_DIR) / "log" / "paper_support"))
    parser.add_argument("--stem", type=str, default="action_mask_schematic")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--show-scene-axes", action="store_true")
    args = parser.parse_args()

    render(
        output_dir=args.output_dir,
        stem=args.stem,
        sample_stride=max(int(args.sample_stride), 1),
        dpi=max(int(args.dpi), 100),
        show_scene_axes=bool(args.show_scene_axes),
    )


if __name__ == "__main__":
    main()
