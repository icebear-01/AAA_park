import argparse
import heapq
import json
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
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
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as PolygonPatch

from configs import NUM_STEP, STEP_LENGTH, VALID_SPEED, VALID_STEER, VehicleBox, WHEEL_BASE
import env.car_parking_base as car_parking_base_module
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.task_utils import clone_map
from env.vehicle import KSModel, State, Status
from model.agent.build_utils import resolve_agent_init_configs
from model.agent.sac_agent import SACAgent as SAC

from export_vector_case_plots import (
    PALETTE,
    draw_map,
    generate_scene_map,
    map_to_dict,
    reset_env_from_map,
    state_box_coords,
    state_to_dict,
)


SCENE_SPECS = [
    {
        "key": "complex_bay",
        "level": "Complex",
        "map_case_id": 0,
        "label": "Complex Bay",
        "orientation": "vertical",
        "candidate_seeds": [22, 55, 10, 1634, 2601, 2567, 2974, 2, 7, 12, 17, 31, 42, 58, 73, 88, 119],
    },
    {
        "key": "complex_parallel",
        "level": "Complex",
        "map_case_id": 1,
        "label": "Complex Parallel",
        "orientation": "horizontal",
        "candidate_seeds": [9, 13, 99, 2755, 94, 26, 68, 20, 31, 44, 61, 77, 103, 128, 151],
    },
    {
        "key": "extrem_parallel",
        "level": "Extrem",
        "map_case_id": 1,
        "label": "Extrem Parallel",
        "orientation": "horizontal_extreme",
        "candidate_seeds": [6263, 41, 1872, 3, 17, 29, 63, 82, 106, 139, 172, 211],
    },
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def angle_wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_abs_diff(a, b):
    return abs(angle_wrap(a - b))


def case_seed_sequence(spec, max_seed):
    seen = set()
    for seed in spec["candidate_seeds"]:
        if seed not in seen:
            seen.add(seed)
            yield seed
    for seed in range(max_seed):
        if seed not in seen:
            seen.add(seed)
            yield seed


def create_pure_rl_env():
    raw_env = CarParking(fps=0, verbose=False, render_mode="rgb_array", enable_rs_assist=False)
    return CarParkingWrapper(raw_env)


def build_sac_agent(ckpt_path, env):
    configs = resolve_agent_init_configs(
        env.observation_shape,
        env.action_space.shape[0],
        ckpt_path=ckpt_path,
    )
    agent = SAC(configs)
    agent.load(ckpt_path, params_only=True)
    return agent


def build_policy_obs(obs, action_mask_input_mode):
    if action_mask_input_mode == "actual" or obs.get("action_mask") is None:
        return obs
    policy_obs = dict(obs)
    fill_value = 1.0 if action_mask_input_mode == "ones" else 0.0
    policy_obs["action_mask"] = np.full_like(
        obs["action_mask"],
        fill_value=fill_value,
        dtype=obs["action_mask"].dtype,
    )
    return policy_obs


def run_pure_rl_case(
    map_obj,
    agent,
    action_seed,
    action_mode,
    action_mask_input_mode="actual",
    terminate_on_collision=False,
):
    seed_everything(action_seed)
    old_env_collide = car_parking_base_module.ENV_COLLIDE
    if terminate_on_collision:
        car_parking_base_module.ENV_COLLIDE = True
    env = create_pure_rl_env()
    env.action_space.seed(action_seed)
    obs = reset_env_from_map(env, map_obj)

    done = False
    step_num = 0
    segments = []
    t0 = time.perf_counter()
    info = {"status": Status.CONTINUE, "path_to_dest": None}

    try:
        while not done:
            step_num += 1
            prev_state = deepcopy(env.unwrapped.vehicle.state)
            policy_obs = build_policy_obs(obs, action_mask_input_mode)
            if action_mode == "raw":
                action, _ = agent.get_action(policy_obs)
            else:
                action, _ = agent.choose_action(policy_obs)
            obs, reward, done, info = env.step(action)
            curr_state = env.unwrapped.vehicle.state
            segments.append(
                {
                    "phase": "single_rl_no_rs",
                    "x0": float(prev_state.loc.x),
                    "y0": float(prev_state.loc.y),
                    "x1": float(curr_state.loc.x),
                    "y1": float(curr_state.loc.y),
                    "action": [float(action[0]), float(action[1])],
                }
            )
            if step_num > 1000:
                break
    finally:
        car_parking_base_module.ENV_COLLIDE = old_env_collide

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = {
        "method": "single_rl_no_rs",
        "status": info["status"].name,
        "final_success": bool(info["status"] == Status.ARRIVED),
        "step_num": int(step_num),
        "inference_time_ms": float(elapsed_ms),
        "avg_step_time_ms": float(elapsed_ms / max(step_num, 1)),
        "trajectory_states": [state_to_dict(state) for state in env.unwrapped.vehicle.trajectory],
        "segments": segments,
        "map": map_to_dict(env.unwrapped.map),
        "rs_assist_used": False,
        "action_mode": action_mode,
        "action_mask_input_mode": action_mask_input_mode,
        "terminate_on_collision": bool(terminate_on_collision),
    }
    env.close()
    return result


@dataclass(order=True)
class QueueNode:
    priority: float
    order: int
    x: float = field(compare=False)
    y: float = field(compare=False)
    yaw: float = field(compare=False)
    g: float = field(compare=False)
    parent: int = field(compare=False, default=-1)
    action: tuple = field(compare=False, default=(0.0, 0.0))
    depth: int = field(compare=False, default=0)


class BoundedHybridAStar:
    def __init__(
        self,
        xy_resolution=0.45,
        yaw_bins=48,
        max_nodes=2200,
        expand_steps=12,
        segment_step_time=4,
        speed=2.0,
    ):
        self.xy_resolution = float(xy_resolution)
        self.yaw_bins = int(yaw_bins)
        self.max_nodes = int(max_nodes)
        self.expand_steps = int(expand_steps)
        self.segment_step_time = int(segment_step_time)
        self.speed = float(speed)
        self.model = KSModel(WHEEL_BASE, STEP_LENGTH, NUM_STEP, VALID_SPEED, VALID_STEER)
        self.env = CarParking(fps=0, verbose=False, render_mode="rgb_array", enable_rs_assist=False)
        self.steers = np.array([-0.75, -0.48, -0.25, 0.0, 0.25, 0.48, 0.75], dtype=np.float64)
        self.directions = np.array([1.0, -1.0], dtype=np.float64)

    def _set_map(self, map_obj):
        self.env.map = clone_map(map_obj)
        self.env.vehicle.reset(self.env.map.start)
        self.env.matrix = self.env.coord_transform_matrix()

    def _key(self, x, y, yaw):
        map_obj = self.env.map
        ix = int(round((x - map_obj.xmin) / self.xy_resolution))
        iy = int(round((y - map_obj.ymin) / self.xy_resolution))
        yaw_norm = (yaw + 2.0 * math.pi) % (2.0 * math.pi)
        itheta = int(round(yaw_norm / (2.0 * math.pi) * self.yaw_bins)) % self.yaw_bins
        return ix, iy, itheta

    def _heuristic(self, x, y, yaw):
        goal = self.env.map.dest
        dist = math.hypot(x - goal.loc.x, y - goal.loc.y)
        heading = angle_abs_diff(yaw, goal.heading)
        return dist + 1.4 * heading

    def _goal_reached(self, x, y, yaw):
        goal = self.env.map.dest
        dist = math.hypot(x - goal.loc.x, y - goal.loc.y)
        heading = angle_abs_diff(yaw, goal.heading)
        return dist < 0.85 and heading < 0.36

    def _rollout(self, node, steer, speed):
        state = State([node.x, node.y, node.yaw, 0.0, 0.0])
        traj = []
        for _ in range(self.expand_steps):
            state = self.model.step(state, [steer, speed], step_time=self.segment_step_time)
            traj.append([float(state.loc.x), float(state.loc.y), float(state.heading)])
        return state, traj

    def search(self, map_obj):
        self._set_map(map_obj)
        start = self.env.map.start
        t0 = time.perf_counter()

        nodes = []
        closed = []
        best_node_idx = 0
        best_h = float("inf")
        visited = set()
        pq = []
        order = 0

        start_h = self._heuristic(start.loc.x, start.loc.y, start.heading)
        start_node = QueueNode(
            priority=start_h,
            order=order,
            x=float(start.loc.x),
            y=float(start.loc.y),
            yaw=float(start.heading),
            g=0.0,
            parent=-1,
            action=(0.0, 0.0),
            depth=0,
        )
        nodes.append(start_node)
        heapq.heappush(pq, start_node)
        visited.add(self._key(start_node.x, start_node.y, start_node.yaw))

        success = False
        goal_idx = -1
        segments = []

        while pq and len(closed) < self.max_nodes:
            current = heapq.heappop(pq)
            current_idx = current.order
            closed.append(current_idx)
            h_value = self._heuristic(current.x, current.y, current.yaw)
            if h_value < best_h:
                best_h = h_value
                best_node_idx = current_idx
            if self._goal_reached(current.x, current.y, current.yaw):
                success = True
                goal_idx = current_idx
                break

            for steer in self.steers:
                for direction in self.directions:
                    speed = float(direction * self.speed)
                    child_state, traj = self._rollout(current, float(steer), speed)
                    if not self.env.is_traj_valid(traj):
                        continue
                    key = self._key(child_state.loc.x, child_state.loc.y, child_state.heading)
                    if key in visited:
                        continue
                    visited.add(key)
                    order += 1
                    step_cost = abs(speed) * STEP_LENGTH * self.segment_step_time * self.expand_steps
                    reverse_penalty = 1.25 if speed < 0.0 else 1.0
                    steer_penalty = 0.20 * abs(float(steer))
                    g = current.g + step_cost * reverse_penalty + steer_penalty
                    priority = g + 1.55 * self._heuristic(child_state.loc.x, child_state.loc.y, child_state.heading)
                    child = QueueNode(
                        priority=priority,
                        order=order,
                        x=float(child_state.loc.x),
                        y=float(child_state.loc.y),
                        yaw=float(child_state.heading),
                        g=float(g),
                        parent=current_idx,
                        action=(float(steer), float(speed)),
                        depth=current.depth + 1,
                    )
                    nodes.append(child)
                    segments.append(
                        {
                            "parent": current_idx,
                            "child": order,
                            "points": [[float(x), float(y), float(yaw)] for x, y, yaw in traj],
                        }
                    )
                    heapq.heappush(pq, child)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        terminal_idx = goal_idx if success else best_node_idx
        best_path = self._reconstruct(nodes, terminal_idx)
        explored = [
            {
                "x": float(nodes[idx].x),
                "y": float(nodes[idx].y),
                "yaw": float(nodes[idx].yaw),
                "g": float(nodes[idx].g),
                "depth": int(nodes[idx].depth),
                "order": int(idx),
            }
            for idx in closed
        ]
        closed_set = set(closed)
        search_nodes = [
            {
                "x": float(node.x),
                "y": float(node.y),
                "yaw": float(node.yaw),
                "g": float(node.g),
                "depth": int(node.depth),
                "order": int(node.order),
                "expanded": bool(node.order in closed_set),
            }
            for node in nodes
        ]

        status = "success" if success else ("node_budget_exhausted" if len(closed) >= self.max_nodes else "open_set_empty")
        result = {
            "method": "bounded_hybrid_astar",
            "status": status,
            "final_success": bool(success),
            "expanded_nodes": int(len(closed)),
            "generated_nodes": int(len(nodes)),
            "max_nodes": int(self.max_nodes),
            "search_time_ms": float(elapsed_ms),
            "xy_resolution": self.xy_resolution,
            "yaw_bins": self.yaw_bins,
            "search_nodes": search_nodes,
            "explored_nodes": explored,
            "best_path": best_path,
            "map": map_to_dict(self.env.map),
        }
        return result

    def _reconstruct(self, nodes, idx):
        path = []
        while idx >= 0 and idx < len(nodes):
            node = nodes[idx]
            path.append({"x": float(node.x), "y": float(node.y), "heading": float(node.yaw)})
            idx = node.parent
        path.reverse()
        return path

    def close(self):
        self.env.close()


def polygon_points_from_map(map_info):
    points = []
    for key in ("start_box", "dest_box"):
        points.extend(map_info[key])
    for obstacle in map_info["obstacles"]:
        points.extend(obstacle)
    return points


def compute_bounds(map_info, extra_points=None, min_width=24.0, min_height=20.0, pad=1.5):
    points = polygon_points_from_map(map_info)
    if extra_points is not None:
        for point in extra_points:
            points.append(point)
    arr = np.asarray(points, dtype=np.float64)
    xmin, ymin = np.min(arr[:, 0]), np.min(arr[:, 1])
    xmax, ymax = np.max(arr[:, 0]), np.max(arr[:, 1])
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    width = max(float(xmax - xmin + 2.0 * pad), float(min_width))
    height = max(float(ymax - ymin + 2.0 * pad), float(min_height))
    return {
        "xmin": float(cx - 0.5 * width),
        "xmax": float(cx + 0.5 * width),
        "ymin": float(cy - 0.5 * height),
        "ymax": float(cy + 0.5 * height),
    }


def apply_bounds(ax, map_info, extra_points=None, show_labels=True):
    map_for_draw = deepcopy(map_info)
    map_for_draw["bounds"] = compute_bounds(map_info, extra_points)
    draw_map(ax, map_for_draw, show_axis_labels=show_labels)
    ax.set_xlabel("x (m)", fontsize=18, labelpad=8, color=PALETTE["dark"])
    ax.set_ylabel("y (m)", fontsize=18, labelpad=10, color=PALETTE["dark"])
    ax.tick_params(labelsize=18, pad=4)
    return map_for_draw["bounds"]


def set_axis_title(ax, title):
    ax.set_title(title, fontsize=18, color=PALETTE["dark"], pad=10, fontweight="semibold")


def draw_hybrid_nodes(ax, hybrid_result, title=None):
    explored = hybrid_result["explored_nodes"]
    generated = hybrid_result.get("search_nodes", explored)
    points = np.array([[node["x"], node["y"]] for node in explored], dtype=np.float64)
    generated_points = np.array([[node["x"], node["y"]] for node in generated], dtype=np.float64)
    extra_points = generated_points.tolist() if len(generated_points) else points.tolist()
    if hybrid_result["best_path"]:
        extra_points.extend([[p["x"], p["y"]] for p in hybrid_result["best_path"]])
    apply_bounds(ax, hybrid_result["map"], extra_points)
    if len(generated_points) > 0:
        ax.scatter(
            generated_points[:, 0],
            generated_points[:, 1],
            s=7.2,
            color="#7393B3",
            alpha=0.24,
            linewidths=0,
            zorder=5,
            rasterized=True,
        )
    if len(points) > 0:
        colors = np.linspace(0.0, 1.0, len(points))
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=colors,
            s=8.5,
            cmap="viridis",
            alpha=0.62,
            linewidths=0,
            zorder=6,
            rasterized=True,
        )
    path = np.array([[p["x"], p["y"]] for p in hybrid_result["best_path"]], dtype=np.float64)
    if len(path) >= 2:
        ax.plot(
            path[:, 0],
            path[:, 1],
            color="#111827",
            linewidth=2.0,
            alpha=0.82,
            zorder=9,
            solid_capstyle="round",
        )
    if title:
        set_axis_title(ax, title)
    add_failure_note(
        ax,
        "search failed\n"
        f"{hybrid_result['expanded_nodes']} expanded / {hybrid_result['generated_nodes']} generated",
    )


def draw_rl_trajectory(ax, rl_result, title=None):
    states = rl_result["trajectory_states"]
    points = np.array([[state["x"], state["y"]] for state in states], dtype=np.float64)
    apply_bounds(ax, rl_result["map"], points.tolist())
    if len(points) >= 2:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap="inferno",
            norm=plt.Normalize(0, max(len(segments), 1)),
            linewidths=3.0,
            alpha=0.96,
            zorder=8,
            capstyle="round",
            joinstyle="round",
        )
        lc.set_array(np.arange(len(segments)))
        ax.add_collection(lc)
        ax.plot(points[:, 0], points[:, 1], color="white", linewidth=5.0, alpha=0.55, zorder=7)

    stride = max(1, int(math.ceil(len(states) / 16)))
    footprint_indices = list(range(0, len(states), stride))
    if footprint_indices and footprint_indices[-1] != len(states) - 1:
        footprint_indices.append(len(states) - 1)
    for order_idx, idx in enumerate(footprint_indices):
        state = states[idx]
        t = order_idx / max(len(footprint_indices) - 1, 1)
        face = (0.85, 0.27 + 0.25 * t, 0.12, 0.22 if idx != footprint_indices[-1] else 0.40)
        edge = "#D55E00" if idx != footprint_indices[-1] else "#B91C1C"
        ax.add_patch(
            PolygonPatch(
                state_box_coords(state),
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.15 if idx != footprint_indices[-1] else 1.7,
                zorder=10 + order_idx / max(len(footprint_indices), 1),
            )
        )

    if title:
        set_axis_title(ax, title)
    add_failure_note(ax, f"rollout failed\n{rl_result['status']}, {rl_result['step_num']} steps")


def add_failure_note(ax, text):
    ax.text(
        0.985,
        0.985,
        text,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=12.5,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#E5E7EB", "boxstyle": "round,pad=0.22", "alpha": 0.86},
        zorder=30,
    )


def save_figure(fig, base_path, dpi):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.16)
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def render_case_figures(case, output_dir, dpi):
    key = case["case_key"]
    hybrid_dir = output_dir / "hybrid_astar_nodes"
    rl_dir = output_dir / "pure_rl_trajectories"
    panel_dir = output_dir / "comparison_panels"

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    draw_hybrid_nodes(ax, case["hybrid"], title=f"{case['label']} Hybrid A*")
    save_figure(fig, hybrid_dir / f"{key}_hybrid_nodes", dpi)

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    draw_rl_trajectory(ax, case["pure_rl"], title=f"{case['label']} Pure RL")
    save_figure(fig, rl_dir / f"{key}_pure_rl_trajectory", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), constrained_layout=True)
    draw_hybrid_nodes(axes[0], case["hybrid"], title="Hybrid A* failed search")
    draw_rl_trajectory(axes[1], case["pure_rl"], title="Pure RL failed rollout")
    fig.suptitle(
        f"Path Planning Failure | {case['label']} seed {case['scene_seed']}",
        fontsize=19,
        color=PALETTE["dark"],
        fontweight="semibold",
    )
    save_figure(fig, panel_dir / f"{key}_comparison", dpi)


def render_overview(cases, output_dir, dpi):
    if not cases:
        return
    rows = len(cases)
    fig, axes = plt.subplots(rows, 2, figsize=(13.6, 5.25 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.asarray([axes])
    for row, case in enumerate(cases):
        draw_hybrid_nodes(axes[row, 0], case["hybrid"], title=f"{case['label']} seed {case['scene_seed']} | Hybrid A* failed search")
        draw_rl_trajectory(axes[row, 1], case["pure_rl"], title=f"{case['label']} seed {case['scene_seed']} | Pure RL failed rollout")
    save_figure(fig, output_dir / "overview_failure_efficiency_examples", dpi)


def case_summary_for_manifest(case):
    return {
        "case_key": case["case_key"],
        "label": case["label"],
        "orientation": case["orientation"],
        "level": case["level"],
        "map_case_id": case["map_case_id"],
        "scene_seed": case["scene_seed"],
        "action_seed": case["action_seed"],
        "hybrid_status": case["hybrid"]["status"],
        "hybrid_expanded_nodes": case["hybrid"]["expanded_nodes"],
        "hybrid_generated_nodes": case["hybrid"]["generated_nodes"],
        "hybrid_search_time_ms": case["hybrid"]["search_time_ms"],
        "pure_rl_status": case["pure_rl"]["status"],
        "pure_rl_steps": case["pure_rl"]["step_num"],
        "pure_rl_action_mode": case["pure_rl"]["action_mode"],
    }


def find_cases(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = create_pure_rl_env()
    agent = build_sac_agent(args.forward_ckpt, env)
    env.close()

    planner = BoundedHybridAStar(
        xy_resolution=args.hybrid_xy_resolution,
        yaw_bins=args.hybrid_yaw_bins,
        max_nodes=args.hybrid_max_nodes,
        expand_steps=args.hybrid_expand_steps,
        segment_step_time=args.hybrid_segment_step_time,
        speed=args.hybrid_speed,
    )

    cases = []
    try:
        for spec in SCENE_SPECS:
            picked = 0
            attempts = 0
            print(f"[scan] {spec['label']} target={args.per_spec}")
            for scene_seed in case_seed_sequence(spec, args.max_seed):
                if picked >= args.per_spec:
                    break
                attempts += 1
                try:
                    map_obj = generate_scene_map(spec["level"], scene_seed, spec["map_case_id"])
                except Exception as exc:
                    print(f"[skip] {spec['key']} seed={scene_seed} map error: {exc}")
                    continue

                hybrid = planner.search(map_obj)
                if hybrid["final_success"] or hybrid["expanded_nodes"] < args.min_hybrid_nodes:
                    continue

                action_seed = int(args.action_seed_base + scene_seed + 10000 * spec["map_case_id"])
                pure_rl = run_pure_rl_case(map_obj, agent, action_seed, args.rl_action_mode)
                if pure_rl["final_success"] or pure_rl["step_num"] < args.min_rl_steps:
                    continue

                case_idx = picked + 1
                case_key = f"{spec['key']}_seed{scene_seed}_case{case_idx}"
                case = {
                    "case_key": case_key,
                    "label": spec["label"],
                    "orientation": spec["orientation"],
                    "level": spec["level"],
                    "map_case_id": spec["map_case_id"],
                    "scene_seed": int(scene_seed),
                    "action_seed": int(action_seed),
                    "hybrid": hybrid,
                    "pure_rl": pure_rl,
                }
                cases.append(case)
                picked += 1
                render_case_figures(case, output_dir, args.png_dpi)
                print(
                    "[picked] "
                    f"{case_key}: hybrid={hybrid['status']} nodes={hybrid['expanded_nodes']} "
                    f"rl={pure_rl['status']} steps={pure_rl['step_num']}"
                )
            print(f"[scan] {spec['label']} picked={picked} attempts={attempts}")
    finally:
        planner.close()

    return cases


def write_manifest(cases, args):
    output_dir = Path(args.output_dir).resolve()
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "forward_ckpt": str(Path(args.forward_ckpt).resolve()),
        "output_dir": str(output_dir),
        "note": "Extrem bay is not defined by generate_scene_map in this benchmark; vertical examples use Complex Bay.",
        "args": vars(args),
        "cases": [case_summary_for_manifest(case) for case in cases],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Export Hybrid A* and pure-RL failure efficiency examples.")
    parser.add_argument("--forward-ckpt", default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--output-dir", default="src/log/paper_support/failure_efficiency_examples")
    parser.add_argument("--per-spec", type=int, default=2)
    parser.add_argument("--max-seed", type=int, default=300)
    parser.add_argument("--action-seed-base", type=int, default=91000)
    parser.add_argument("--rl-action-mode", choices=["choose", "raw"], default="choose")
    parser.add_argument("--min-rl-steps", type=int, default=55)
    parser.add_argument("--hybrid-max-nodes", type=int, default=1800)
    parser.add_argument("--min-hybrid-nodes", type=int, default=900)
    parser.add_argument("--hybrid-xy-resolution", type=float, default=0.45)
    parser.add_argument("--hybrid-yaw-bins", type=int, default=48)
    parser.add_argument("--hybrid-expand-steps", type=int, default=12)
    parser.add_argument("--hybrid-segment-step-time", type=int, default=4)
    parser.add_argument("--hybrid-speed", type=float, default=2.0)
    parser.add_argument("--png-dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    cases = find_cases(args)
    output_dir = Path(args.output_dir).resolve()
    render_overview(cases, output_dir, args.png_dpi)
    write_manifest(cases, args)
    print(f"[done] output_dir={output_dir}")
    print(f"[done] cases={len(cases)}")


if __name__ == "__main__":
    main()
