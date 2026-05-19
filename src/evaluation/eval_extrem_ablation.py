import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
from tqdm import tqdm

from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import Status, VALID_SPEED
from model.agent.build_utils import resolve_agent_init_configs
from model.agent.bidirectional_parking_agent import (
    BidirectionalParkingAgent,
    capture_global_rng_state,
    restore_global_rng_state,
)
from model.agent.parking_agent import RsPlanner
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC


DEFAULT_FORWARD_CKPT = os.path.join(ROOT_DIR, "model", "ckpt", "HOPE_PPO.pt")
DEFAULT_UNPARK_CKPT = os.path.join(
    ROOT_DIR,
    "log",
    "exp",
    "unpark_ppo_20260323_120610",
    "PPO_unpark_best.pt",
)


MODE_SPECS = {
    "ppo_raw": {
        "dual_model": False,
        "use_rs_assist": False,
        "use_action_mask_exec": False,
        "use_action_mask_input": False,
        "description": "PPO single model, no RS, no action-mask post process, no action-mask input.",
    },
    "ppo_plus_rs": {
        "dual_model": False,
        "use_rs_assist": True,
        "use_action_mask_exec": False,
        "use_action_mask_input": False,
        "description": "PPO single model + RS assist, still without action-mask post process/input.",
    },
    "ppo_plus_rs_plus_mask_exec": {
        "dual_model": False,
        "use_rs_assist": True,
        "use_action_mask_exec": True,
        "use_action_mask_input": False,
        "description": "PPO single model + RS assist + action-mask post process; mask input is constantized.",
    },
    "ppo_plus_rs_plus_mask_exec_input": {
        "dual_model": False,
        "use_rs_assist": True,
        "use_action_mask_exec": True,
        "use_action_mask_input": True,
        "description": "PPO single model + RS assist + action-mask post process + true mask input.",
    },
    "ppo_full_dual": {
        "dual_model": True,
        "use_rs_assist": True,
        "use_action_mask_exec": True,
        "use_action_mask_input": True,
        "description": "PPO forward model + unpark PPO branch + RS/mask full bidirectional inference.",
    },
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def chunk_list(items, n_chunks):
    n_chunks = max(1, min(n_chunks, len(items)))
    buckets = [[] for _ in range(n_chunks)]
    for idx, item in enumerate(items):
        buckets[idx % n_chunks].append(item)
    return [bucket for bucket in buckets if bucket]


def build_env(enable_rs_assist):
    raw_env = CarParking(
        fps=0,
        verbose=False,
        render_mode="rgb_array",
        use_action_mask=True,
        enable_rs_assist=enable_rs_assist,
    )
    env = CarParkingWrapper(raw_env)
    return env


def build_rl_agent(ckpt_path, env):
    agent_type = PPO if "ppo" in ckpt_path.lower() else SAC
    configs = resolve_agent_init_configs(
        env.observation_shape,
        env.action_space.shape[0],
        ckpt_path=ckpt_path,
    )
    rl_agent = agent_type(configs)
    rl_agent.load(ckpt_path, params_only=True)
    return rl_agent


class AblationParkingAgent(object):
    def __init__(
        self,
        rl_agent,
        planner=None,
        use_action_mask_exec=True,
        use_action_mask_input=True,
        action_mask_fill_value=1.0,
    ):
        self.agent = rl_agent
        self.planner = planner
        self.use_action_mask_exec = use_action_mask_exec
        self.use_action_mask_input = use_action_mask_input
        self.action_mask_fill_value = action_mask_fill_value
        self.last_action_source = "policy"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)

    def reset(self):
        self.last_action_source = "policy"
        if self.planner is not None:
            self.planner.reset()

    def set_planner_path(self, path=None, forced=False):
        if self.planner is None:
            return
        if path is not None and (forced or self.planner.route is None):
            self.planner.set_rs_path(path)

    @property
    def executing_rs(self):
        return not (self.planner is None or self.planner.route is None)

    def _build_policy_obs(self, obs):
        if self.use_action_mask_input or obs.get("action_mask") is None:
            return obs
        policy_obs = dict(obs)
        policy_obs["action_mask"] = np.full_like(
            obs["action_mask"],
            fill_value=self.action_mask_fill_value,
            dtype=obs["action_mask"].dtype,
        )
        return policy_obs

    def _sample_policy_action(self, obs):
        policy_obs = self._build_policy_obs(obs)
        dist = self.agent._actor_forward(policy_obs)
        action_mask = obs.get("action_mask") if self.use_action_mask_exec else None
        return self.agent._post_process_action(dist, action_mask)

    def get_log_prob(self, obs, action):
        policy_obs = self._build_policy_obs(obs)
        return self.agent.get_log_prob(policy_obs, action)

    def choose_action(self, obs):
        if not self.executing_rs:
            self.last_action_source = "policy"
            return self._sample_policy_action(obs)
        self.last_action_source = "rs_assist"
        action = np.asarray(self.planner.get_action(), dtype=np.float32)
        log_prob = self.get_log_prob(obs, action)
        return action, log_prob

    def get_action(self, obs):
        return self.choose_action(obs)


def build_single_runner(env, mode_spec, forward_ckpt):
    rl_agent = build_rl_agent(forward_ckpt, env)
    planner = None
    if mode_spec["use_rs_assist"]:
        step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
        planner = RsPlanner(step_ratio)
    return AblationParkingAgent(
        rl_agent,
        planner=planner,
        use_action_mask_exec=mode_spec["use_action_mask_exec"],
        use_action_mask_input=mode_spec["use_action_mask_input"],
    )


def build_dual_runner(env, mode_spec, forward_ckpt, unpark_ckpt):
    step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
    forward_agent = AblationParkingAgent(
        build_rl_agent(forward_ckpt, env),
        planner=RsPlanner(step_ratio) if mode_spec["use_rs_assist"] else None,
        use_action_mask_exec=mode_spec["use_action_mask_exec"],
        use_action_mask_input=mode_spec["use_action_mask_input"],
    )
    forward_rng_state = capture_global_rng_state()
    unpark_agent = AblationParkingAgent(
        build_rl_agent(unpark_ckpt, env),
        planner=RsPlanner(step_ratio),
        use_action_mask_exec=mode_spec["use_action_mask_exec"],
        use_action_mask_input=mode_spec["use_action_mask_input"],
    )
    restore_global_rng_state(forward_rng_state)
    return BidirectionalParkingAgent(forward_agent, unpark_agent)


def evaluate_episode(env, agent, mode_spec, episode_id):
    obs = env.reset(episode_id, None, "Extrem")
    if mode_spec["dual_model"]:
        agent.reset(env)
    else:
        agent.reset()

    done = False
    step_num = 0
    path_to_dest_non_null_steps = 0
    path_to_dest_non_null_episode = False
    rs_exec_used = False
    connection_used = False
    connection_step = None

    while not done:
        step_num += 1
        if mode_spec["dual_model"]:
            prev_connection = agent.connection_used
            action, _ = agent.choose_action(obs, env)
            if agent.last_action_phase in {"forward_rs_assist", "rs_connector"}:
                rs_exec_used = True
            if (not prev_connection) and agent.connection_used and connection_step is None:
                connection_step = step_num
        else:
            if agent.executing_rs:
                rs_exec_used = True
            action, _ = agent.choose_action(obs)

        next_obs, reward, done, info = env.step(action)
        if info["path_to_dest"] is not None:
            path_to_dest_non_null_steps += 1
            path_to_dest_non_null_episode = True
            if mode_spec["dual_model"]:
                if not agent.connection_used:
                    agent.forward_agent.set_planner_path(info["path_to_dest"])
            else:
                agent.set_planner_path(info["path_to_dest"])

        if mode_spec["dual_model"] and agent.connection_used:
            connection_used = True
        obs = next_obs

    result = {
        "episode_id": int(episode_id),
        "success": int(info["status"] == Status.ARRIVED),
        "status": info["status"].name,
        "step_num": int(step_num),
        "path_to_dest_non_null_steps": int(path_to_dest_non_null_steps),
        "path_to_dest_non_null_episode": int(path_to_dest_non_null_episode),
        "rs_exec_used": int(rs_exec_used),
    }
    if mode_spec["dual_model"]:
        result["connection_used"] = int(connection_used)
        result["connection_step"] = None if connection_step is None else int(connection_step)
        result["planning_success"] = int(connection_used or info["status"] == Status.ARRIVED)
    return result


def worker_run(worker_id, episode_ids, mode_name, forward_ckpt, unpark_ckpt, progress=False):
    mode_spec = MODE_SPECS[mode_name]
    seed = 1000 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = build_env(mode_spec["use_rs_assist"])
    env.action_space.seed(seed)
    if mode_spec["dual_model"]:
        agent = build_dual_runner(env, mode_spec, forward_ckpt, unpark_ckpt)
    else:
        agent = build_single_runner(env, mode_spec, forward_ckpt)

    try:
        iterator = tqdm(episode_ids, desc="worker-%s" % worker_id, leave=False) if progress else episode_ids
        results = [evaluate_episode(env, agent, mode_spec, episode_id) for episode_id in iterator]
        return results
    finally:
        env.close()


def aggregate_results(mode_name, forward_ckpt, unpark_ckpt, num_workers, all_results, wall_time_sec):
    mode_spec = MODE_SPECS[mode_name]
    all_results = sorted(all_results, key=lambda item: item["episode_id"])
    success_count = sum(item["success"] for item in all_results)
    success_steps = [item["step_num"] for item in all_results if item["success"]]
    status_count = Counter(item["status"] for item in all_results)
    summary = {
        "mode": mode_name,
        "description": mode_spec["description"],
        "checkpoint": forward_ckpt,
        "unpark_checkpoint": unpark_ckpt if mode_spec["dual_model"] else None,
        "policy": "PPO" if "ppo" in os.path.basename(forward_ckpt).lower() else "SAC",
        "level": "Extrem",
        "eval_episode_target": len(all_results),
        "episodes_completed": len(all_results),
        "success_definition": "status == ARRIVED",
        "dual_model": bool(mode_spec["dual_model"]),
        "use_rs_assist": bool(mode_spec["use_rs_assist"]),
        "use_action_mask_exec": bool(mode_spec["use_action_mask_exec"]),
        "use_action_mask_input": bool(mode_spec["use_action_mask_input"]),
        "action_mask_input_disabled_as": "constant_ones" if not mode_spec["use_action_mask_input"] else "actual_mask",
        "num_workers": int(num_workers),
        "success_rate": float(success_count / max(len(all_results), 1)),
        "success_count": int(success_count),
        "avg_step_num": float(np.mean([item["step_num"] for item in all_results])) if all_results else math.nan,
        "avg_success_step_num": float(np.mean(success_steps)) if success_steps else math.nan,
        "path_to_dest_non_null_steps": int(sum(item["path_to_dest_non_null_steps"] for item in all_results)),
        "path_to_dest_non_null_episodes": int(sum(item["path_to_dest_non_null_episode"] for item in all_results)),
        "rs_exec_episodes": int(sum(item["rs_exec_used"] for item in all_results)),
        "status_count": dict(status_count),
        "wall_time_sec": float(wall_time_sec),
        "final": True,
    }
    if mode_spec["dual_model"]:
        summary["connection_rate"] = float(
            sum(item["connection_used"] for item in all_results) / max(len(all_results), 1)
        )
        summary["planning_success_rate"] = float(
            sum(item["planning_success"] for item in all_results) / max(len(all_results), 1)
        )
        summary["final_parking_success_rate"] = summary["success_rate"]
    return summary


def save_outputs(save_dir, summary, episode_results):
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "summary.json")
    details_path = os.path.join(save_dir, "episode_details.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(details_path, "w") as f:
        json.dump(episode_results, f, indent=2)
    return summary_path, details_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=sorted(MODE_SPECS.keys()))
    parser.add_argument("--forward_ckpt", type=str, default=DEFAULT_FORWARD_CKPT)
    parser.add_argument("--unpark_ckpt", type=str, default=DEFAULT_UNPARK_CKPT)
    parser.add_argument("--eval_episode", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--start_method", type=str, default="spawn", choices=["spawn", "fork"])
    parser.add_argument("--show_worker_progress", type=str2bool, default=False)
    args = parser.parse_args()

    episode_ids = list(range(1, args.eval_episode + 1))
    chunks = chunk_list(episode_ids, args.num_workers)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, "log", "eval", "%s_%s" % (args.mode, timestamp))

    start_time = time.time()
    ctx = get_context(args.start_method)
    with ctx.Pool(len(chunks)) as pool:
        async_results = [
            pool.apply_async(
                worker_run,
                kwds={
                    "worker_id": worker_id,
                    "episode_ids": chunk,
                    "mode_name": args.mode,
                    "forward_ckpt": args.forward_ckpt,
                    "unpark_ckpt": args.unpark_ckpt,
                    "progress": args.show_worker_progress,
                },
            )
            for worker_id, chunk in enumerate(chunks)
        ]
        episode_results = []
        for async_result in tqdm(async_results, desc="Collect", dynamic_ncols=True):
            episode_results.extend(async_result.get())

    wall_time_sec = time.time() - start_time
    summary = aggregate_results(
        mode_name=args.mode,
        forward_ckpt=args.forward_ckpt,
        unpark_ckpt=args.unpark_ckpt,
        num_workers=len(chunks),
        all_results=episode_results,
        wall_time_sec=wall_time_sec,
    )
    summary_path, details_path = save_outputs(save_dir, summary, episode_results)

    print(json.dumps(summary, indent=2))
    print("summary_path:", summary_path)
    print("details_path:", details_path)


if __name__ == "__main__":
    main()
