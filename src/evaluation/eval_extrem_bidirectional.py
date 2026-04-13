import sys
import time
import os
import json
import argparse
from collections import defaultdict

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

import numpy as np
import torch
from tqdm import trange

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC
from model.agent.build_utils import resolve_agent_init_configs
from model.agent.parking_agent import ParkingAgent, RsPlanner
from model.agent.bidirectional_parking_agent import (
    BidirectionalParkingAgent,
    capture_global_rng_state,
    restore_global_rng_state,
)
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED, Status
from configs import *


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def build_agent(ckpt_path, env):
    agent_type = PPO if 'ppo' in ckpt_path.lower() else SAC
    configs = resolve_agent_init_configs(
        env.observation_shape,
        env.action_space.shape[0],
        ckpt_path=ckpt_path,
    )
    rl_agent = agent_type(configs)
    rl_agent.load(ckpt_path, params_only=True)
    step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
    return ParkingAgent(rl_agent, RsPlanner(step_ratio))


PHASE_NAMES = (
    "forward_policy",
    "forward_rs_assist",
    "rs_connector",
    "reverse_replay",
    "forward_policy_after_connection",
)


def sign_of_speed(action):
    speed = float(action[1])
    if speed > 1e-6:
        return 1
    if speed < -1e-6:
        return -1
    return 0


def init_phase_dict(value=0.0):
    return {phase: value for phase in PHASE_NAMES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("forward_ckpt", type=str)
    parser.add_argument("unpark_ckpt", type=str)
    parser.add_argument("--eval_episode", type=int, default=100)
    parser.add_argument("--verbose", type=str2bool, default=True)
    parser.add_argument("--visualize", type=str2bool, default=False)
    args = parser.parse_args()

    verbose = args.verbose
    if args.visualize:
        raw_env = CarParking(fps=100, verbose=verbose)
    else:
        raw_env = CarParking(fps=0, verbose=verbose, render_mode="rgb_array")
    env = CarParkingWrapper(raw_env)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path = os.path.join(ROOT_DIR, 'log', 'eval', 'bidirectional_extrem_%s' % timestamp)
    os.makedirs(save_path, exist_ok=True)

    env.action_space.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    forward_agent = build_agent(args.forward_ckpt, env)
    forward_rng_state = capture_global_rng_state()
    unpark_agent = build_agent(args.unpark_ckpt, env)
    bidirectional_agent = BidirectionalParkingAgent(forward_agent, unpark_agent)
    restore_global_rng_state(forward_rng_state)

    succ_record = []
    connection_record = []
    step_record = []
    final_succ_record = []
    total_path_length_record = []
    total_gear_shift_record = []
    phase_step_totals = defaultdict(float)
    phase_path_length_totals = defaultdict(float)
    phase_gear_shift_totals = defaultdict(float)
    phase_usage_totals = defaultdict(int)
    episode_details = []

    episode_iter = trange(args.eval_episode, desc="Eval Extrem", dynamic_ncols=True) if not verbose else range(args.eval_episode)
    for i in episode_iter:
        obs = env.reset(i + 1, None, 'Extrem')
        bidirectional_agent.reset(env)
        done = False
        step_num = 0
        prev_speed_sign = None
        total_path_length = 0.0
        total_gear_shifts = 0
        phase_steps = init_phase_dict(0)
        phase_path_lengths = init_phase_dict(0.0)
        phase_gear_shifts = init_phase_dict(0)

        while not done:
            step_num += 1
            prev_loc = env.unwrapped.vehicle.state.loc
            action, _ = bidirectional_agent.choose_action(obs, env)
            phase = bidirectional_agent.last_action_phase
            phase_steps.setdefault(phase, 0)
            phase_path_lengths.setdefault(phase, 0.0)
            phase_gear_shifts.setdefault(phase, 0)
            speed_sign = sign_of_speed(action)
            if prev_speed_sign is not None and speed_sign != 0 and prev_speed_sign != 0 and speed_sign != prev_speed_sign:
                total_gear_shifts += 1
                phase_gear_shifts[phase] += 1
            if speed_sign != 0:
                prev_speed_sign = speed_sign
            next_obs, reward, done, info = env.step(action)
            curr_loc = env.unwrapped.vehicle.state.loc
            path_length = prev_loc.distance(curr_loc)
            total_path_length += path_length
            phase_steps[phase] += 1
            phase_path_lengths[phase] += path_length
            obs = next_obs
            if info['path_to_dest'] is not None and not bidirectional_agent.connection_used:
                bidirectional_agent.forward_agent.set_planner_path(info['path_to_dest'])

        planning_success = int(bidirectional_agent.connection_used)
        final_success = int(info['status'] == Status.ARRIVED)
        succ_record.append(planning_success)
        final_succ_record.append(final_success)
        connection_record.append(int(bidirectional_agent.connection_used))
        step_record.append(step_num)
        total_path_length_record.append(total_path_length)
        total_gear_shift_record.append(total_gear_shifts)
        for phase in PHASE_NAMES:
            phase_step_totals[phase] += phase_steps.get(phase, 0)
            phase_path_length_totals[phase] += phase_path_lengths.get(phase, 0.0)
            phase_gear_shift_totals[phase] += phase_gear_shifts.get(phase, 0)
            phase_usage_totals[phase] += int(phase_steps.get(phase, 0) > 0)

        episode_details.append({
            "episode_id": i + 1,
            "planning_success": planning_success,
            "final_parking_success": final_success,
            "status": info['status'].name,
            "connection_used": bool(bidirectional_agent.connection_used),
            "connection_index": bidirectional_agent.connection_index,
            "step_num": step_num,
            "total_path_length": total_path_length,
            "total_gear_shifts": total_gear_shifts,
            "phase_steps": phase_steps,
            "phase_path_lengths": phase_path_lengths,
            "phase_gear_shifts": phase_gear_shifts,
        })

        if verbose:
            print(
                "episode %s status=%s planning_success=%s final_success=%s connection=%s step_num=%s" % (
                    i + 1,
                    info['status'].name,
                    succ_record[-1],
                    final_succ_record[-1],
                    connection_record[-1],
                    step_num,
                )
            )
        else:
            episode_iter.set_postfix(
                planning="%.3f" % float(np.mean(succ_record)),
                final="%.3f" % float(np.mean(final_succ_record)),
            )

    result = {
        "forward_ckpt": args.forward_ckpt,
        "unpark_ckpt": args.unpark_ckpt,
        "eval_episode": args.eval_episode,
        "success_rate": float(np.mean(succ_record)),
        "planning_success_rate": float(np.mean(succ_record)),
        "connection_rate": float(np.mean(connection_record)),
        "final_parking_success_rate": float(np.mean(final_succ_record)),
        "direct_final_success_rate": float(np.mean([
            int(item["final_parking_success"] and not item["connection_used"])
            for item in episode_details
        ])),
        "avg_step_num": float(np.mean(step_record)),
        "avg_total_path_length": float(np.mean(total_path_length_record)),
        "avg_total_gear_shifts": float(np.mean(total_gear_shift_record)),
        "avg_phase_steps": {phase: float(phase_step_totals[phase] / args.eval_episode) for phase in PHASE_NAMES},
        "avg_phase_path_lengths": {phase: float(phase_path_length_totals[phase] / args.eval_episode) for phase in PHASE_NAMES},
        "avg_phase_gear_shifts": {phase: float(phase_gear_shift_totals[phase] / args.eval_episode) for phase in PHASE_NAMES},
        "phase_usage_rate": {phase: float(phase_usage_totals[phase] / args.eval_episode) for phase in PHASE_NAMES},
    }
    with open(os.path.join(save_path, 'summary.json'), 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(save_path, 'episode_details.json'), 'w') as f:
        json.dump(episode_details, f, indent=2)

    print(json.dumps(result, indent=2))
    env.close()
