import os
import random
from multiprocessing import get_context

import numpy as np


def _configure_headless():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")


def _worker(remote, parent_remote, env_kwargs, seed):
    parent_remote.close()
    _configure_headless()

    from env.car_parking_out_base import CarParkingOut
    from env.env_wrapper import CarParkingWrapper
    from env.vehicle import VALID_SPEED

    np.random.seed(seed)
    random.seed(seed)

    raw_env = CarParkingOut(**env_kwargs)
    env = CarParkingWrapper(raw_env)
    env.action_space.seed(seed)

    step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(env.reset(*data))
            elif cmd == "step":
                remote.send(env.step(data))
            elif cmd == "get_env_info":
                remote.send({
                    "observation_shape": env.observation_shape,
                    "action_dim": env.action_space.shape[0],
                    "step_ratio": step_ratio,
                })
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise ValueError(f"Unknown worker command: {cmd}")
    finally:
        env.close()


class ParallelCarParkingOutEnv:
    def __init__(self, num_envs, env_kwargs=None, base_seed=0, start_method="spawn"):
        if num_envs < 1:
            raise ValueError("`num_envs` must be >= 1")

        self.num_envs = num_envs
        self.env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
        self.base_seed = base_seed
        self.start_method = start_method
        self.closed = False

        ctx = get_context(start_method)
        self.remotes = []
        self.processes = []
        for worker_id in range(num_envs):
            remote, child_remote = ctx.Pipe()
            process = ctx.Process(
                target=_worker,
                args=(child_remote, remote, self.env_kwargs, self.base_seed + worker_id),
                daemon=True,
            )
            process.start()
            child_remote.close()
            self.remotes.append(remote)
            self.processes.append(process)

        self.remotes[0].send(("get_env_info", None))
        env_info = self.remotes[0].recv()
        self.observation_shape = env_info["observation_shape"]
        self.action_dim = env_info["action_dim"]
        self.step_ratio = env_info["step_ratio"]

    def reset_all(self, reset_args_list):
        if len(reset_args_list) != self.num_envs:
            raise ValueError("`reset_args_list` length must match `num_envs`")
        for remote, reset_args in zip(self.remotes, reset_args_list):
            remote.send(("reset", reset_args))
        return [remote.recv() for remote in self.remotes]

    def reset_one(self, env_idx, reset_args):
        self.remotes[env_idx].send(("reset", reset_args))
        return self.remotes[env_idx].recv()

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError("`actions` length must match `num_envs`")
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True
