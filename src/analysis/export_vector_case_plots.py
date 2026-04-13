import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
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
import torch
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch

from configs import *
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.parking_map_dlp import ParkingMapDLP
from env.map_base import Area
from env.parking_map_normal import (
    ParkingMapNormal,
    generate_bay_parking_case,
    generate_parallel_parking_case,
)
from env.task_utils import clone_map
from env.vehicle import State, Status, VALID_SPEED
from model.agent.bidirectional_parking_agent import (
    BidirectionalParkingAgent,
    capture_global_rng_state,
    restore_global_rng_state,
)
from model.agent.build_utils import resolve_agent_init_configs
from model.agent.parking_agent import ParkingAgent, RsPlanner
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC


DEFAULT_SCENES = [
    {"level": "Normal", "scene_seed": 3, "map_case_id": 0, "tag": "normal_seed3"},
    {"level": "Complex", "scene_seed": 2, "map_case_id": 0, "tag": "complex_seed2"},
    {"level": "Extrem", "scene_seed": 1, "map_case_id": 0, "tag": "extrem_seed1"},
    {"level": "Extrem", "scene_seed": 3, "map_case_id": 0, "tag": "extrem_seed3"},
]

PALETTE = {
    "bg": "#FBFBF8",
    "grid": "#D6DADF",
    "dark": "#223044",
    "obstacle": "#B8BFC8",
    "start_fill": "#D5E4F3",
    "start_edge": "#557A9B",
    "dest_fill": "#DDEBD3",
    "dest_edge": "#648A46",
    "forward_policy": "#355070",
    "forward_rs_assist": "#5FA8D3",
    "rs_connector": "#D98F2B",
    "reverse_replay": "#A06CD5",
    "forward_policy_after_connection": "#C8553D",
    "reverse_branch": "#7B6FD6",
    "anchor": "#111827",
    "success": "#2A9D8F",
    "fail": "#C8553D",
}

UNIFIED_ROUTE_COLOR = "#355070"

PHASE_ORDER = [
    "forward_policy",
    "forward_rs_assist",
    "rs_connector",
    "reverse_replay",
    "forward_policy_after_connection",
]

DISPLAY_CORNER_ROUNDING_PASSES = 4
DISPLAY_TRAJECTORY_SMOOTH_PASSES = 2
DISPLAY_DENSE_SAMPLE_SPACING = 0.10
DISPLAY_RAW_TERMINAL_POINTS = 4
DISPLAY_FOOTPRINT_STRIDE = 3
DISPLAY_FOOTPRINT_LINE_COLOR = "#F1D400"
DISPLAY_FOOTPRINT_START_COLOR = "#24C7B8"
DISPLAY_FOOTPRINT_END_COLOR = "#2F6CF6"
DISPLAY_FOOTPRINT_EDGE = "#13CFE3"


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_env():
    raw_env = CarParking(fps=0, verbose=False, render_mode="rgb_array")
    return CarParkingWrapper(raw_env)


def build_agent(ckpt_path, env):
    agent_type = PPO if "ppo" in ckpt_path.lower() else SAC
    configs = resolve_agent_init_configs(
        env.observation_shape,
        env.action_space.shape[0],
        ckpt_path=ckpt_path,
    )
    rl_agent = agent_type(configs)
    rl_agent.load(ckpt_path, params_only=True)
    step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
    return ParkingAgent(rl_agent, RsPlanner(step_ratio))


def policy_action(agent, obs):
    agent_name = agent.agent.__class__.__name__.lower() if hasattr(agent, "agent") else agent.__class__.__name__.lower()
    if "ppo" in agent_name:
        return agent.choose_action(obs)
    return agent.get_action(obs)


def state_to_dict(state):
    return {
        "x": float(state.loc.x),
        "y": float(state.loc.y),
        "heading": float(state.heading),
        "speed": float(getattr(state, "speed", 0.0)),
        "steering": float(getattr(state, "steering", 0.0)),
    }


def ring_coords(shape_or_area):
    shape = shape_or_area.shape if hasattr(shape_or_area, "shape") else shape_or_area
    coords = np.asarray(shape.coords)
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def map_to_dict(map_obj):
    return {
        "map_level": getattr(map_obj, "map_level", None),
        "bounds": {
            "xmin": float(map_obj.xmin),
            "xmax": float(map_obj.xmax),
            "ymin": float(map_obj.ymin),
            "ymax": float(map_obj.ymax),
        },
        "start": state_to_dict(map_obj.start),
        "dest": state_to_dict(map_obj.dest),
        "start_box": ring_coords(map_obj.start_box).tolist(),
        "dest_box": ring_coords(map_obj.dest_box).tolist(),
        "obstacles": [ring_coords(obst).tolist() for obst in map_obj.obstacles],
    }


def reset_env_from_map(env, map_obj):
    unwrapped = env.unwrapped
    unwrapped.reward = 0.0
    unwrapped.prev_reward = 0.0
    unwrapped.accum_arrive_reward = 0.0
    unwrapped.t = 0.0
    unwrapped.map = clone_map(map_obj)
    unwrapped.vehicle.reset(unwrapped.map.start)
    unwrapped.matrix = unwrapped.coord_transform_matrix()
    return env.obs_func(unwrapped.step()[0])


def build_parking_map(level, map_case_id, start, dest, obstacles):
    parking_map = ParkingMapNormal(level)
    parking_map.case_id = map_case_id
    parking_map.start = State(list(start) + [0, 0])
    parking_map.start_box = parking_map.start.create_box()
    parking_map.dest = State(list(dest) + [0, 0])
    parking_map.dest_box = parking_map.dest.create_box()
    parking_map.xmin = np.floor(min(parking_map.start.loc.x, parking_map.dest.loc.x) - 10)
    parking_map.xmax = np.ceil(max(parking_map.start.loc.x, parking_map.dest.loc.x) + 10)
    parking_map.ymin = np.floor(min(parking_map.start.loc.y, parking_map.dest.loc.y) - 10)
    parking_map.ymax = np.ceil(max(parking_map.start.loc.y, parking_map.dest.loc.y) + 10)
    parking_map.obstacles = [
        Area(shape=obs, subtype="obstacle", color=(150, 150, 150, 255))
        for obs in obstacles
    ]
    parking_map.n_obstacle = len(parking_map.obstacles)
    return clone_map(parking_map)


def generate_scene_map(level, scene_seed, map_case_id):
    seed_everything(scene_seed)
    if level in {"Normal", "Complex", "Extrem"}:
        if level == "Extrem" and map_case_id == 0:
            raise ValueError("Extrem bay scenes are not defined in the current benchmark.")
        if map_case_id == 0:
            start, dest, obstacles = generate_bay_parking_case(level)
        else:
            start, dest, obstacles = generate_parallel_parking_case(level)
        return build_parking_map(level, map_case_id, start, dest, obstacles)
    if level == "dlp":
        parking_map = ParkingMapDLP()
        parking_map.reset(map_case_id, None)
        return clone_map(parking_map)
    raise ValueError(f"Unsupported level: {level}")


def run_single_case(map_obj, ckpt_path, action_seed):
    seed_everything(action_seed)
    env = create_env()
    env.action_space.seed(action_seed)
    obs = reset_env_from_map(env, map_obj)

    agent = build_agent(ckpt_path, env)
    agent.reset()

    done = False
    step_num = 0
    rs_assist_used = False
    segments = []
    phase_counts = {phase: 0 for phase in PHASE_ORDER}

    while not done:
        step_num += 1
        phase = "forward_rs_assist" if agent.executing_rs else "forward_policy"
        prev_state = deepcopy(env.unwrapped.vehicle.state)
        action, _ = policy_action(agent, obs)
        obs, reward, done, info = env.step(action)
        curr_state = env.unwrapped.vehicle.state
        segments.append(
            {
                "phase": phase,
                "x0": float(prev_state.loc.x),
                "y0": float(prev_state.loc.y),
                "x1": float(curr_state.loc.x),
                "y1": float(curr_state.loc.y),
                "action": [float(action[0]), float(action[1])],
            }
        )
        phase_counts[phase] += 1
        if info["path_to_dest"] is not None:
            rs_assist_used = True
            agent.set_planner_path(info["path_to_dest"])

    result = {
        "method": "single",
        "status": info["status"].name,
        "final_success": bool(info["status"] == Status.ARRIVED),
        "planning_success": bool(info["status"] == Status.ARRIVED),
        "step_num": step_num,
        "rs_assist_used": rs_assist_used,
        "trajectory_states": [state_to_dict(state) for state in env.unwrapped.vehicle.trajectory],
        "segments": segments,
        "phase_steps": phase_counts,
        "map": map_to_dict(env.unwrapped.map),
    }
    env.close()
    return result


def run_dual_case(map_obj, forward_ckpt, unpark_ckpt, action_seed):
    seed_everything(action_seed)
    env = create_env()
    env.action_space.seed(action_seed)
    obs = reset_env_from_map(env, map_obj)

    forward_agent = build_agent(forward_ckpt, env)
    forward_rng_state = capture_global_rng_state()
    unpark_agent = build_agent(unpark_ckpt, env)
    agent = BidirectionalParkingAgent(forward_agent, unpark_agent)
    agent.reset(env)
    restore_global_rng_state(forward_rng_state)

    done = False
    step_num = 0
    connection_step = None
    segments = []
    phase_counts = {phase: 0 for phase in PHASE_ORDER}

    while not done:
        step_num += 1
        prev_connection_used = agent.connection_used
        prev_state = deepcopy(env.unwrapped.vehicle.state)
        action, _ = agent.choose_action(obs, env)
        phase = agent.last_action_phase
        if (not prev_connection_used) and agent.connection_used and connection_step is None:
            connection_step = step_num
        obs, reward, done, info = env.step(action)
        curr_state = env.unwrapped.vehicle.state
        segments.append(
            {
                "phase": phase,
                "x0": float(prev_state.loc.x),
                "y0": float(prev_state.loc.y),
                "x1": float(curr_state.loc.x),
                "y1": float(curr_state.loc.y),
                "action": [float(action[0]), float(action[1])],
            }
        )
        phase_counts.setdefault(phase, 0)
        phase_counts[phase] += 1
        if info["path_to_dest"] is not None and not agent.connection_used:
            agent.forward_agent.set_planner_path(info["path_to_dest"])

    result = {
        "method": "dual",
        "status": info["status"].name,
        "final_success": bool(info["status"] == Status.ARRIVED),
        "planning_success": bool(agent.connection_used or info["status"] == Status.ARRIVED),
        "connection_used": bool(agent.connection_used),
        "connection_index": agent.connection_index,
        "connection_step": connection_step,
        "step_num": step_num,
        "trajectory_states": [state_to_dict(state) for state in env.unwrapped.vehicle.trajectory],
        "segments": segments,
        "phase_steps": phase_counts,
        "reverse_states": [state_to_dict(state) for state in agent.reverse_states],
        "reverse_actions": [action.tolist() for action in agent.reverse_actions],
        "connector_path": None if agent.connection_path is None else {
            "x": [float(x) for x in agent.connection_path.x],
            "y": [float(y) for y in agent.connection_path.y],
            "yaw": [float(yaw) for yaw in agent.connection_path.yaw],
            "lengths": [float(v) for v in agent.connection_path.lengths],
            "ctypes": list(agent.connection_path.ctypes),
        },
        "map": map_to_dict(env.unwrapped.map),
    }
    env.close()
    return result


def dedupe_points(points, tol=1e-8):
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    deduped = [np.asarray(points[0], dtype=np.float64)]
    for point in points[1:]:
        point = np.asarray(point, dtype=np.float64)
        if np.linalg.norm(point - deduped[-1]) > tol:
            deduped.append(point)
    return np.vstack(deduped)


def smooth_polyline(points, iterations=DISPLAY_TRAJECTORY_SMOOTH_PASSES):
    points = dedupe_points(points)
    if len(points) < 3 or iterations <= 0:
        return points
    smoothed = points.copy()
    for _ in range(iterations):
        if len(smoothed) < 3:
            break
        new_points = [smoothed[0]]
        for idx in range(len(smoothed) - 1):
            p0 = smoothed[idx]
            p1 = smoothed[idx + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            if idx == 0:
                new_points.append(q)
            else:
                new_points.extend([q, r])
        new_points.extend([0.25 * smoothed[-2] + 0.75 * smoothed[-1], smoothed[-1]])
        smoothed = dedupe_points(new_points)
    return smoothed


def densify_polyline(points, spacing=DISPLAY_DENSE_SAMPLE_SPACING):
    points = dedupe_points(points)
    if len(points) < 2:
        return points
    dense_points = [points[0]]
    for idx in range(len(points) - 1):
        p0 = points[idx]
        p1 = points[idx + 1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-8:
            continue
        n_parts = max(int(np.ceil(seg_len / spacing)), 1)
        for part in range(1, n_parts + 1):
            t = part / n_parts
            dense_points.append((1.0 - t) * p0 + t * p1)
    return dedupe_points(dense_points)


def smooth_route(points):
    points = dedupe_points(points)
    if len(points) < 3:
        return points

    # Keep the terminal approach unsmoothed so the display curve never
    # appears to overshoot the final pose.
    tail_keep = min(DISPLAY_RAW_TERMINAL_POINTS, len(points) - 1)
    split_idx = len(points) - tail_keep
    smooth_controls = points[: split_idx + 1]
    raw_tail_controls = points[split_idx:]

    # Paper figures use a stronger display-only rounding pass so the
    # trajectory looks closer to a continuous parking path while still
    # preserving the true start/end states.
    rounded_controls = smooth_polyline(smooth_controls, iterations=DISPLAY_CORNER_ROUNDING_PASSES)
    rounded_controls[0] = smooth_controls[0]
    rounded_controls[-1] = smooth_controls[-1]

    dense_prefix = densify_polyline(rounded_controls, spacing=DISPLAY_DENSE_SAMPLE_SPACING)
    smoothed_prefix = smooth_polyline(dense_prefix, iterations=DISPLAY_TRAJECTORY_SMOOTH_PASSES)
    if len(smoothed_prefix) >= 2:
        smoothed_prefix[0] = smooth_controls[0]
        smoothed_prefix[-1] = smooth_controls[-1]

    dense_tail = densify_polyline(raw_tail_controls, spacing=DISPLAY_DENSE_SAMPLE_SPACING)
    if len(smoothed_prefix) == 0:
        merged = dense_tail
    elif len(dense_tail) == 0:
        merged = smoothed_prefix
    else:
        merged = np.vstack([smoothed_prefix[:-1], dense_tail])
    if len(merged) >= 2:
        merged[0] = points[0]
        merged[-1] = points[-1]
    return dedupe_points(merged)


def lerp_color(color0, color1, t):
    c0 = np.asarray(to_rgb(color0), dtype=np.float64)
    c1 = np.asarray(to_rgb(color1), dtype=np.float64)
    return tuple((1.0 - t) * c0 + t * c1)


def state_box_coords(state_dict):
    coords = np.asarray(VehicleBox.coords, dtype=np.float64)
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    cos_theta = np.cos(state_dict["heading"])
    sin_theta = np.sin(state_dict["heading"])
    rot = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float64)
    shifted = coords @ rot.T
    shifted[:, 0] += state_dict["x"]
    shifted[:, 1] += state_dict["y"]
    return shifted


def sampled_trajectory_indices(traj_states, stride=DISPLAY_FOOTPRINT_STRIDE):
    if not traj_states:
        return []
    indices = list(range(0, len(traj_states), max(int(stride), 1)))
    if indices[-1] != len(traj_states) - 1:
        indices.append(len(traj_states) - 1)
    return indices


def draw_footprint_trajectory(ax, traj_states, zorder_base=6, stride=DISPLAY_FOOTPRINT_STRIDE):
    trajectory_points = np.array([[state["x"], state["y"]] for state in traj_states], dtype=np.float64)
    smoothed_trajectory = smooth_route(trajectory_points)
    if len(smoothed_trajectory) >= 2:
        ax.plot(
            smoothed_trajectory[:, 0],
            smoothed_trajectory[:, 1],
            color=DISPLAY_FOOTPRINT_LINE_COLOR,
            linewidth=1.55,
            alpha=0.98,
            zorder=zorder_base + 2,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    sampled_idx = sampled_trajectory_indices(traj_states, stride=stride)
    n = max(len(sampled_idx) - 1, 1)
    for order_idx, idx in enumerate(sampled_idx):
        state = traj_states[idx]
        t = order_idx / n
        fill_color = lerp_color(DISPLAY_FOOTPRINT_START_COLOR, DISPLAY_FOOTPRINT_END_COLOR, t)
        alpha = 0.34 if order_idx < len(sampled_idx) - 1 else 0.54
        linewidth = 1.05 if order_idx < len(sampled_idx) - 1 else 1.2
        ax.add_patch(
            PolygonPatch(
                state_box_coords(state),
                closed=True,
                facecolor=fill_color,
                edgecolor=DISPLAY_FOOTPRINT_EDGE,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder_base + order_idx / max(len(sampled_idx), 1),
            )
        )


def phase_paths_from_segments(segments):
    paths = []
    current_phase = None
    current_points = []

    def flush():
        nonlocal current_phase, current_points
        if current_phase is not None and len(current_points) >= 2:
            paths.append({"phase": current_phase, "points": dedupe_points(current_points)})
        current_phase = None
        current_points = []

    for seg in segments:
        p0 = np.array([seg["x0"], seg["y0"]], dtype=np.float64)
        p1 = np.array([seg["x1"], seg["y1"]], dtype=np.float64)
        if np.allclose(p0, p1):
            continue
        if seg["phase"] != current_phase:
            flush()
            current_phase = seg["phase"]
            current_points = [p0, p1]
            continue
        current_points.append(p1)
    flush()
    return paths


def plot_path(ax, points, color, linewidth, zorder, linestyle="-", alpha=1.0, smooth=True):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return
    if smooth:
        points = smooth_route(points)
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
        solid_joinstyle="round",
        dash_capstyle="round",
        dash_joinstyle="round",
    )


def choose_tick_step(span):
    if span <= 12:
        return 2.0
    if span <= 24:
        return 5.0
    return 10.0


def add_corner_axes(ax, bounds):
    x_span = bounds["xmax"] - bounds["xmin"]
    y_span = bounds["ymax"] - bounds["ymin"]
    scale = min(x_span, y_span) * 0.09
    origin_x = bounds["xmin"] + 0.08 * x_span
    origin_y = bounds["ymin"] + 0.10 * y_span
    ax.annotate(
        "",
        xy=(origin_x + scale, origin_y),
        xytext=(origin_x, origin_y),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["dark"], lw=1.0, shrinkA=0, shrinkB=0, alpha=0.8),
        zorder=20,
    )
    ax.annotate(
        "",
        xy=(origin_x, origin_y + scale),
        xytext=(origin_x, origin_y),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["dark"], lw=1.0, shrinkA=0, shrinkB=0, alpha=0.8),
        zorder=20,
    )


def catmull_rom_segment(points, idx, samples=10):
    n_points = len(points)
    if idx < 0 or idx >= n_points - 1:
        return np.empty((0, 2), dtype=np.float64)
    p1 = points[idx]
    p2 = points[idx + 1]
    if np.allclose(p1, p2):
        return np.empty((0, 2), dtype=np.float64)
    p0 = points[idx - 1] if idx > 0 else (2.0 * p1 - p2)
    p3 = points[idx + 2] if idx + 2 < n_points else (2.0 * p2 - p1)
    t_values = np.linspace(0.0, 1.0, samples, endpoint=(idx == n_points - 2))
    t = t_values[:, None]
    curve = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * (t ** 2)
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * (t ** 3)
    )
    return curve


def build_smoothed_segment_curves(traj_states, segments, samples=10):
    points = np.array([[state["x"], state["y"]] for state in traj_states], dtype=np.float64)
    curves = []
    for idx, seg in enumerate(segments):
        curve = catmull_rom_segment(points, idx, samples=samples)
        if len(curve) < 2:
            continue
        curves.append({"phase": seg["phase"], "points": curve})
    return curves


def merge_curves(curves):
    merged = []
    for curve in curves:
        points = curve["points"]
        if len(points) == 0:
            continue
        if not merged:
            merged.extend(points)
            continue
        if np.allclose(merged[-1], points[0]):
            merged.extend(points[1:])
        else:
            merged.extend(points)
    if not merged:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(merged, dtype=np.float64)


def group_phase_curves(segment_curves):
    phase_curves = []
    current_phase = None
    current_curves = []

    def flush():
        nonlocal current_phase, current_curves
        if current_phase is not None and current_curves:
            phase_curves.append({"phase": current_phase, "points": merge_curves(current_curves)})
        current_phase = None
        current_curves = []

    for curve in segment_curves:
        if curve["phase"] != current_phase:
            flush()
            current_phase = curve["phase"]
        current_curves.append(curve)
    flush()
    return phase_curves


def draw_map(ax, map_info, show_axis_labels=False):
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(True, linestyle="--", linewidth=0.55, color=PALETTE["grid"], alpha=0.45)
    for obstacle in map_info["obstacles"]:
        ax.add_patch(
            PolygonPatch(
                obstacle,
                closed=True,
                facecolor=PALETTE["obstacle"],
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
        )
    ax.add_patch(
        PolygonPatch(
            map_info["dest_box"],
            closed=True,
            facecolor=PALETTE["dest_fill"],
            edgecolor=PALETTE["dest_edge"],
            linewidth=1.3,
            alpha=0.9,
            zorder=3,
        )
    )
    ax.add_patch(
        PolygonPatch(
            map_info["start_box"],
            closed=True,
            facecolor=PALETTE["start_fill"],
            edgecolor=PALETTE["start_edge"],
            linewidth=1.3,
            linestyle="--",
            alpha=0.95,
            zorder=4,
        )
    )
    bounds = map_info["bounds"]
    pad = 1.5
    ax.set_xlim(bounds["xmin"] - pad, bounds["xmax"] + pad)
    ax.set_ylim(bounds["ymin"] - pad, bounds["ymax"] + pad)
    ax.set_aspect("equal", adjustable="box")
    x_step = choose_tick_step(bounds["xmax"] - bounds["xmin"])
    y_step = choose_tick_step(bounds["ymax"] - bounds["ymin"])
    x_start = x_step * np.floor(bounds["xmin"] / x_step)
    y_start = y_step * np.floor(bounds["ymin"] / y_step)
    ax.set_xticks(np.arange(x_start, bounds["xmax"] + x_step, x_step))
    ax.set_yticks(np.arange(y_start, bounds["ymax"] + y_step, y_step))
    ax.tick_params(length=2.6, width=0.8, labelsize=16.0, colors=PALETTE["dark"])
    for spine_name, spine in ax.spines.items():
        if spine_name in ("left", "bottom"):
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color(PALETTE["grid"])
        else:
            spine.set_visible(False)


def draw_case(ax, scene_label, result, show_legend=False, show_axis_labels=False, trajectory_style="line", footprint_stride=DISPLAY_FOOTPRINT_STRIDE):
    draw_map(ax, result["map"], show_axis_labels=show_axis_labels)

    traj_states = result["trajectory_states"]
    if trajectory_style == "footprints":
        draw_footprint_trajectory(ax, traj_states, stride=footprint_stride)
    else:
        trajectory_points = np.array([[state["x"], state["y"]] for state in traj_states], dtype=np.float64)
        smoothed_trajectory = smooth_route(trajectory_points)
        if len(smoothed_trajectory) >= 2:
            plot_path(
                ax,
                smoothed_trajectory,
                color="white",
                linewidth=4.4,
                alpha=0.95,
                zorder=7,
                smooth=False,
            )
            plot_path(
                ax,
                smoothed_trajectory,
                color=UNIFIED_ROUTE_COLOR,
                linewidth=2.7,
                alpha=0.98,
                zorder=8,
                smooth=False,
            )

    start_state = traj_states[0]
    end_state = traj_states[-1]
    end_color = PALETTE["success"] if result["final_success"] else PALETTE["fail"]
    ax.scatter(start_state["x"], start_state["y"], s=42, color=PALETTE["start_edge"], edgecolor="white", linewidth=0.8, zorder=10)
    ax.scatter(end_state["x"], end_state["y"], s=42, color=end_color, edgecolor="white", linewidth=0.8, zorder=10)

    status_text = "ARRIVED" if result["final_success"] else result["status"]
    method_label = "Single HOPE" if result["method"] == "single" else "Dual HOPE"
    conn_text = ""
    if result["method"] == "dual":
        conn_text = f", conn={int(result['connection_used'])}"
        if result.get("connection_index") is not None:
            conn_text += f", idx={result['connection_index']}"
    ax.set_title(f"{scene_label} | {method_label}\n{status_text}, steps={result['step_num']}{conn_text}", fontsize=11.0, color=PALETTE["dark"])

    if show_legend:
        if trajectory_style == "footprints":
            handles = [
                Line2D([0], [0], color=DISPLAY_FOOTPRINT_LINE_COLOR, lw=1.6, label="path"),
                PolygonPatch([[0, 0], [1, 0], [1, 0.5], [0, 0.5]], closed=True, facecolor=DISPLAY_FOOTPRINT_START_COLOR, edgecolor=DISPLAY_FOOTPRINT_EDGE, linewidth=1.0, label="vehicle footprint"),
            ]
        else:
            handles = [
                Line2D([0], [0], color=UNIFIED_ROUTE_COLOR, lw=2.7, label="trajectory"),
            ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=1, frameon=False, fontsize=9.2)


def save_case_figure(result, scene_label, out_path, png_dpi, trajectory_style="line", footprint_stride=DISPLAY_FOOTPRINT_STRIDE):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    draw_case(ax, scene_label, result, show_legend=True, show_axis_labels=True, trajectory_style=trajectory_style, footprint_stride=footprint_stride)
    fig.savefig(out_path.with_suffix(".png"), dpi=png_dpi)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def save_panel(results, out_dir, png_dpi, trajectory_style="line", footprint_stride=DISPLAY_FOOTPRINT_STRIDE):
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(13.8, 4.6 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    for row_idx, item in enumerate(results):
        draw_case(
            axes[row_idx, 0],
            item["label"],
            item["single"],
            show_legend=(row_idx == 0),
            show_axis_labels=(row_idx == nrows - 1),
            trajectory_style=trajectory_style,
            footprint_stride=footprint_stride,
        )
        draw_case(
            axes[row_idx, 1],
            item["label"],
            item["dual"],
            show_legend=False,
            show_axis_labels=(row_idx == nrows - 1),
            trajectory_style=trajectory_style,
            footprint_stride=footprint_stride,
        )
    fig.suptitle("Vector Trajectory Replays from Raw States", fontsize=15, y=0.996, color=PALETTE["dark"])
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    panel_path = out_dir / "vector_case_panel"
    fig.savefig(panel_path.with_suffix(".png"), dpi=png_dpi)
    fig.savefig(panel_path.with_suffix(".pdf"))
    fig.savefig(panel_path.with_suffix(".svg"))
    plt.close(fig)


def parse_scene_spec(text):
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid scene spec '{text}'. Use 'Level:seed', e.g. 'Normal:3'.")
    level, seed_text = parts
    return {
        "level": level,
        "scene_seed": int(seed_text),
        "map_case_id": 0,
        "tag": f"{level.lower()}_seed{seed_text}",
    }


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward_ckpt", type=str, default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--unpark_ckpt", type=str, default="src/log/exp/unpark_ppo_20260323_120610/PPO_unpark_best.pt")
    parser.add_argument("--scene", action="append", default=None, help="Generated scene spec as 'Level:seed', e.g. 'Extrem:3'")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--png_dpi", type=int, default=240)
    parser.add_argument("--verbose", type=str2bool, default=True)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = Path(args.save_dir) if args.save_dir else Path(ROOT_DIR) / "log" / "analysis" / f"vector_case_plots_{timestamp}"
    ensure_dir(save_dir)

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

    scene_specs = DEFAULT_SCENES if not args.scene else [parse_scene_spec(text) for text in args.scene]
    index = {
        "forward_ckpt": args.forward_ckpt,
        "unpark_ckpt": args.unpark_ckpt,
        "scenes": [],
    }
    panel_inputs = []

    for scene_idx, spec in enumerate(scene_specs):
        label = f"{spec['level']} seed {spec['scene_seed']}"
        if args.verbose:
            print(f"[{scene_idx + 1}/{len(scene_specs)}] replay {label}")
        map_obj = generate_scene_map(spec["level"], spec["scene_seed"], spec["map_case_id"])
        action_seed = spec["scene_seed"] + 1000
        single_result = run_single_case(map_obj, args.forward_ckpt, action_seed)
        dual_result = run_dual_case(map_obj, args.forward_ckpt, args.unpark_ckpt, action_seed)

        scene_dir = save_dir / spec["tag"]
        ensure_dir(scene_dir)
        (scene_dir / "scene_spec.json").write_text(json.dumps(spec, indent=2))
        (scene_dir / "scene_map.json").write_text(json.dumps(map_to_dict(map_obj), indent=2))
        (scene_dir / "single_trace.json").write_text(json.dumps(single_result, indent=2))
        (scene_dir / "dual_trace.json").write_text(json.dumps(dual_result, indent=2))

        save_case_figure(single_result, label, scene_dir / "single_vector", png_dpi=args.png_dpi)
        save_case_figure(dual_result, label, scene_dir / "dual_vector", png_dpi=args.png_dpi)

        scene_summary = {
            "label": label,
            "tag": spec["tag"],
            "scene_seed": spec["scene_seed"],
            "level": spec["level"],
            "single": {
                "status": single_result["status"],
                "final_success": single_result["final_success"],
                "step_num": single_result["step_num"],
                "rs_assist_used": single_result["rs_assist_used"],
            },
            "dual": {
                "status": dual_result["status"],
                "planning_success": dual_result["planning_success"],
                "final_success": dual_result["final_success"],
                "connection_used": dual_result["connection_used"],
                "connection_index": dual_result["connection_index"],
                "connection_step": dual_result["connection_step"],
                "step_num": dual_result["step_num"],
            },
        }
        index["scenes"].append(scene_summary)
        panel_inputs.append({"label": label, "single": single_result, "dual": dual_result})

    save_panel(panel_inputs, save_dir, png_dpi=args.png_dpi)
    (save_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
