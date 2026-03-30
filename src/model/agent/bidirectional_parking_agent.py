import copy
import random
from collections import deque

import numpy as np
import torch

from env.car_parking_out_base import CarParkingOut
from env.env_wrapper import CarParkingWrapper
from configs import (
    UNPARK_CONNECT_ATTEMPT_INTERVAL,
    UNPARK_CONNECT_DISABLE_NEAR_GOAL_DIST,
    UNPARK_CONNECT_MIN_FORWARD_STEPS,
    UNPARK_CONNECT_STRIDE,
    UNPARK_FORWARD_PROGRESS_MIN_DELTA,
    UNPARK_FORWARD_PROGRESS_WINDOW,
    UNPARK_PREPARE_STEPS,
    UNPARK_STALL_CONFIRM_WINDOWS,
    UNPARK_STALL_MAX_NET_PROGRESS_RATIO,
    UNPARK_STALL_MIN_TRAVEL,
    UNPARK_STALL_PROGRESS_MAX_DELTA,
)


class BidirectionalParkingAgent(object):
    def __init__(
        self,
        forward_agent,
        unpark_agent,
        reverse_rollout_steps: int = UNPARK_PREPARE_STEPS,
        connect_stride: int = UNPARK_CONNECT_STRIDE,
        connect_attempt_interval: int = UNPARK_CONNECT_ATTEMPT_INTERVAL,
        forward_progress_window: int = UNPARK_FORWARD_PROGRESS_WINDOW,
        forward_progress_min_delta: float = UNPARK_FORWARD_PROGRESS_MIN_DELTA,
        min_forward_steps_before_connect: int = UNPARK_CONNECT_MIN_FORWARD_STEPS,
        disable_connect_near_goal_dist: float = UNPARK_CONNECT_DISABLE_NEAR_GOAL_DIST,
        stall_progress_max_delta: float = UNPARK_STALL_PROGRESS_MAX_DELTA,
        stall_min_travel: float = UNPARK_STALL_MIN_TRAVEL,
        stall_max_net_progress_ratio: float = UNPARK_STALL_MAX_NET_PROGRESS_RATIO,
        stall_confirm_windows: int = UNPARK_STALL_CONFIRM_WINDOWS,
    ) -> None:
        self.forward_agent = forward_agent
        self.unpark_agent = unpark_agent
        self.reverse_rollout_steps = reverse_rollout_steps
        self.connect_stride = max(1, connect_stride)
        self.connect_attempt_interval = max(1, connect_attempt_interval)
        self.forward_progress_window = max(2, forward_progress_window)
        self.forward_progress_min_delta = max(0.0, forward_progress_min_delta)
        self.min_forward_steps_before_connect = max(0, min_forward_steps_before_connect)
        self.disable_connect_near_goal_dist = max(0.0, disable_connect_near_goal_dist)
        self.stall_progress_max_delta = max(0.0, stall_progress_max_delta)
        self.stall_min_travel = max(0.0, stall_min_travel)
        self.stall_max_net_progress_ratio = float(np.clip(stall_max_net_progress_ratio, 0.0, 1.0))
        self.stall_confirm_windows = max(1, stall_confirm_windows)
        self.reverse_env = None
        self.reverse_states = []
        self.reverse_actions = []
        self.connector_actions = []
        self.connection_index = None
        self.connection_path = None
        self.connection_used = False
        self.forward_step_count = 0
        self.forward_dist_history = deque(maxlen=self.forward_progress_window)
        self.forward_pos_history = deque(maxlen=self.forward_progress_window)
        self.stall_window_count = 0
        self.last_action_phase = "forward_policy"

    def _policy_action(self, agent, obs):
        agent_name = agent.agent.__class__.__name__.lower() if hasattr(agent, "agent") else agent.__class__.__name__.lower()
        if "ppo" in agent_name:
            return agent.choose_action(obs)
        return agent.get_action(obs)

    def reset(self, env):
        self.forward_agent.reset()
        self.unpark_agent.reset()
        self.reverse_states = []
        self.reverse_actions = []
        self.connector_actions = []
        self.connection_index = None
        self.connection_path = None
        self.connection_used = False
        self.forward_step_count = 0
        self.forward_dist_history.clear()
        self.forward_pos_history.clear()
        self.stall_window_count = 0
        self.last_action_phase = "forward_policy"
        self._prepare_reverse_branch(env)

    def _build_reverse_env(self, env):
        raw_env = CarParkingOut(
            fps=0,
            verbose=False,
            render_mode="rgb_array",
            use_lidar_observation=env.unwrapped.use_lidar_observation,
            use_img_observation=env.unwrapped.use_img_observation,
            use_action_mask=env.unwrapped.use_action_mask,
        )
        reverse_env = CarParkingWrapper(raw_env)
        raw_obs = reverse_env.unwrapped.reset_from_map(env.unwrapped.map)
        obs = reverse_env.obs_func(raw_obs)
        return reverse_env, obs

    def _prepare_reverse_branch(self, env):
        # Isolate reverse-branch sampling from the forward policy RNG stream.
        # Without this, the unparking rollout advances global RNG state and can
        # change the forward policy trajectory even when no connection is used.
        rng_state = capture_global_rng_state()

        try:
            self.reverse_env, obs = self._build_reverse_env(env)
            self.reverse_states.append(copy.deepcopy(self.reverse_env.unwrapped.vehicle.state))

            for _ in range(self.reverse_rollout_steps):
                action, _ = self._policy_action(self.unpark_agent, obs)
                next_obs, reward, done, info = self.reverse_env.step(action)
                self.reverse_actions.append(np.array(action, dtype=np.float32))
                self.reverse_states.append(copy.deepcopy(self.reverse_env.unwrapped.vehicle.state))
                obs = next_obs
                if done:
                    break
        finally:
            restore_global_rng_state(rng_state)

    def _candidate_indices(self):
        if not self.reverse_states:
            return []
        indices = list(range(len(self.reverse_states) - 1, -1, -self.connect_stride))
        if indices[-1] != 0:
            indices.append(0)
        return indices

    def _build_connector_actions(self, state_index):
        connector_actions = []
        for action in reversed(self.reverse_actions[:state_index]):
            steer, speed = action
            connector_actions.append(np.array([steer, -speed], dtype=np.float32))
        return connector_actions

    def _distance_to_dest(self, env):
        return env.unwrapped.vehicle.state.loc.distance(env.unwrapped.map.dest.loc)

    def _record_forward_progress(self, env):
        self.forward_step_count += 1
        self.forward_dist_history.append(self._distance_to_dest(env))
        state = env.unwrapped.vehicle.state
        self.forward_pos_history.append(np.array([state.loc.x, state.loc.y], dtype=np.float64))

    def _forward_progressing_well(self):
        if len(self.forward_dist_history) < self.forward_progress_window:
            return False

        history = list(self.forward_dist_history)
        progress_delta = history[0] - history[-1]
        if progress_delta < self.forward_progress_min_delta:
            return False

        return all(history[i] >= history[i + 1] - 1e-6 for i in range(len(history) - 1))

    def _forward_stalled(self):
        if len(self.forward_dist_history) < self.forward_progress_window:
            return False

        dist_history = list(self.forward_dist_history)
        pos_history = list(self.forward_pos_history)
        if len(pos_history) < 2:
            return False

        progress_delta = dist_history[0] - dist_history[-1]
        if progress_delta > self.stall_progress_max_delta:
            return False

        travel_len = 0.0
        for idx in range(len(pos_history) - 1):
            travel_len += float(np.linalg.norm(pos_history[idx + 1] - pos_history[idx]))
        if travel_len < self.stall_min_travel:
            return False

        net_progress = float(np.linalg.norm(pos_history[-1] - pos_history[0]))
        progress_ratio = net_progress / max(travel_len, 1e-6)
        return progress_ratio <= self.stall_max_net_progress_ratio

    def _update_stall_state(self):
        if self._forward_stalled():
            self.stall_window_count += 1
        else:
            self.stall_window_count = 0

    def _should_try_connect(self, env):
        if self.connection_used:
            return False
        if self.forward_step_count < self.min_forward_steps_before_connect:
            return False
        if self._distance_to_dest(env) <= self.disable_connect_near_goal_dist:
            return False
        if (self.forward_step_count - 1) % self.connect_attempt_interval != 0:
            return False
        if self._forward_progressing_well():
            return False
        if self.stall_window_count < self.stall_confirm_windows:
            return False
        return True

    def _try_connect(self, env):
        if self.connection_used:
            return False

        for state_index in self._candidate_indices():
            rs_path = env.unwrapped.find_rs_path_to_state(self.reverse_states[state_index])
            if rs_path is None:
                continue

            self.forward_agent.set_planner_path(rs_path, forced=True)
            self.connector_actions = self._build_connector_actions(state_index)
            self.connection_index = state_index
            self.connection_path = rs_path
            self.connection_used = True
            return True

        return False

    def choose_action(self, obs, env):
        # After the forward RS connector is consumed, replay the reverse
        # unparking rollout backwards to drive back into the slot.
        if self.connector_actions and not self.forward_agent.executing_rs:
            self.last_action_phase = "reverse_replay"
            action = self.connector_actions.pop(0)
            log_prob = self.forward_agent.get_log_prob(obs, action)
            return action, log_prob

        if not self.forward_agent.executing_rs and not self.connector_actions:
            self._record_forward_progress(env)
            self._update_stall_state()
            if self._should_try_connect(env):
                self._try_connect(env)

        action, log_prob = self._policy_action(self.forward_agent, obs)
        if self.forward_agent.executing_rs:
            self.last_action_phase = "rs_connector" if self.connection_used else "forward_rs_assist"
        elif self.connection_used:
            self.last_action_phase = "forward_policy_after_connection"
        else:
            self.last_action_phase = "forward_policy"
        return action, log_prob


def capture_global_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_global_rng_state(rng_state):
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.random.set_rng_state(rng_state["torch"])
    if rng_state["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
