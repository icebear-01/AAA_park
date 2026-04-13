import sys
import time
import os
from shutil import copyfile
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for path in [ROOT_DIR, CURRENT_DIR]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from model.agent.build_utils import (
    clone_network_configs,
    configure_unpark_img_mode,
    infer_unpark_img_mode_from_ckpt,
    normalize_unpark_img_mode,
    resolve_agent_init_configs,
)
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent
from env.car_parking_out_base import CarParkingOut
from env.env_wrapper import CarParkingWrapper
from env.parallel_env import ParallelCarParkingOutEnv
from env.vehicle import Status
from evaluation.eval_utils import eval
from configs import *


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_levels(levels_arg):
    levels = [level.strip() for level in levels_arg.split(",") if level.strip()]
    if not levels:
        raise ValueError("`--levels` must contain at least one level")
    return levels


class LevelChoose():
    def __init__(self, levels) -> None:
        self.levels = levels
        self.level_to_idx = {level: idx for idx, level in enumerate(levels)}
        self.success_record = {level: [] for level in levels}
        self.scene_record = []
        self.history_horizon = 200

    def choose_level(self):
        if len(self.scene_record) < self.history_horizon:
            level = self._choose_uniform()
        else:
            if np.random.random() > 0.5:
                level = self._choose_worst_perform()
            else:
                level = self._choose_uniform()
        self.scene_record.append(level)
        return level

    def update_success_record(self, success: int, level: str = None):
        if level is None:
            level = self.scene_record[-1]
        self.success_record[level].append(success)

    def _choose_uniform(self):
        case_count = np.zeros(len(self.levels))
        for i in range(min(len(self.scene_record), self.history_horizon)):
            idx = self.level_to_idx[self.scene_record[-(i + 1)]]
            case_count[idx] += 1
        return self.levels[int(np.argmin(case_count))]

    def _choose_worst_perform(self):
        success_rate = []
        for level in self.levels:
            recent_success_record = self.success_record[level][-min(250, len(self.success_record[level])):]
            if len(recent_success_record) == 0:
                success_rate.append(0.0)
            else:
                success_rate.append(np.mean(recent_success_record))
        fail_rate = 1.0 - np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.01, 1.0)
        fail_rate = fail_rate / np.sum(fail_rate)
        return self.levels[int(np.random.choice(np.arange(len(fail_rate)), p=fail_rate))]

    def recent_success_rates(self, window=100):
        rates = {}
        for level in self.levels:
            recent_success_record = self.success_record[level][-min(window, len(self.success_record[level])):]
            if len(recent_success_record) == 0:
                rates[level] = 0.0
            else:
                rates[level] = float(np.mean(recent_success_record))
        return rates


def make_save_path():
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path = os.path.join(ROOT_DIR, "log", "exp", f"unpark_ppo_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    return save_path


def resolve_unpark_img_mode(agent_ckpt, img_mode, use_slot_channel):
    normalized_mode = normalize_unpark_img_mode(img_mode, use_slot_channel=use_slot_channel)
    if agent_ckpt is None:
        return normalized_mode

    inferred_mode = infer_unpark_img_mode_from_ckpt(agent_ckpt)
    if inferred_mode is None:
        return normalized_mode
    if inferred_mode != normalized_mode:
        print("override `img_mode` to %s based on checkpoint config" % inferred_mode)
    return inferred_mode


def build_env_kwargs(visualize, verbose, img_mode=UNPARK_IMG_MODE):
    if visualize:
        return {"fps": 100, "verbose": verbose, "img_mode": img_mode}
    return {"fps": 0, "verbose": verbose, "render_mode": "rgb_array", "img_mode": img_mode}


def make_env(visualize, verbose, img_mode=UNPARK_IMG_MODE):
    return CarParkingWrapper(CarParkingOut(**build_env_kwargs(visualize, verbose, img_mode)))


def build_agent(observation_shape, action_dim, args):
    actor_layers, critic_layers = clone_network_configs()
    actor_layers, critic_layers = configure_unpark_img_mode(actor_layers, critic_layers, args.img_mode)
    configs = resolve_agent_init_configs(
        observation_shape,
        action_dim,
        ckpt_path=args.agent_ckpt,
        actor_layers=actor_layers,
        critic_layers=critic_layers,
        extra_configs={"unpark_img_mode": args.img_mode},
    )

    rl_agent = PPO(configs)
    if args.agent_ckpt is not None:
        rl_agent.load(args.agent_ckpt, params_only=True)
        print("load pre-trained model!")
    elif USE_IMG and args.img_ckpt is not None and os.path.exists(args.img_ckpt):
        if args.img_mode != "rgb":
            print("skip loading pretrained image encoder for img_mode=%s" % args.img_mode)
            return rl_agent
        rl_agent.load_img_encoder(args.img_ckpt, require_grad=UPDATE_IMG_ENCODE)
    return rl_agent


def log_episode_metrics(writer, rl_agent, reward_list, reward_per_state_list, recent_level_rates,
                        total_reward, step_num, completed_episodes):
    writer.add_scalar("total_reward", total_reward, completed_episodes)
    writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), completed_episodes)
    writer.add_scalar("step_num", step_num, completed_episodes)
    writer.add_scalar("action_std0", rl_agent.log_std.detach().cpu().numpy().reshape(-1)[0], completed_episodes)
    writer.add_scalar("action_std1", rl_agent.log_std.detach().cpu().numpy().reshape(-1)[1], completed_episodes)
    for level, rate in recent_level_rates.items():
        writer.add_scalar("success_rate_%s" % level, rate, completed_episodes)


def save_reward_curve(save_path, reward_list):
    episodes = [j for j in range(len(reward_list))]
    mean_reward = [np.mean(reward_list[max(0, j - 50):j + 1]) for j in range(len(reward_list))]
    plt.plot(episodes, reward_list)
    plt.plot(episodes, mean_reward)
    plt.xlabel("episodes")
    plt.ylabel("reward")
    fig = plt.gcf()
    fig.savefig("%s/reward.png" % save_path)
    fig.clear()


def save_best_model(rl_agent, save_path, completed_episodes, best_joint_success_rate, recent_level_rates):
    rl_agent.save("%s/PPO_unpark_best.pt" % save_path, params_only=True)
    with open(save_path + "/best.txt", "w") as f_best_log:
        f_best_log.write("episode: %s\n" % completed_episodes)
        f_best_log.write("joint success rate: %s\n" % best_joint_success_rate)
        for level, rate in recent_level_rates.items():
            f_best_log.write("%s success rate: %s\n" % (level, rate))


def maybe_update_agent(rl_agent, writer, update_x):
    if len(rl_agent.memory) >= rl_agent.configs.batch_size:
        actor_loss, critic_loss = rl_agent.update()
        writer.add_scalar("actor_loss", actor_loss, update_x)
        writer.add_scalar("critic_loss", critic_loss, update_x)


def handle_episode_end(rl_agent, writer, save_path, level_chooser, succ_record, reward_list,
                       reward_per_state_list, total_reward, step_num, level_name, success,
                       completed_episodes, best_joint_success_rate, verbose):
    succ_record.append(success)
    level_chooser.update_success_record(success, level_name)
    reward_list.append(total_reward)

    recent_success_rate = float(np.mean(succ_record[-100:]))
    recent_level_rates = level_chooser.recent_success_rates()
    joint_success_rate = min(recent_level_rates.values())

    writer.add_scalar("success_rate_unpark", recent_success_rate, completed_episodes)
    writer.add_scalar("success_rate_joint", joint_success_rate, completed_episodes)
    log_episode_metrics(
        writer,
        rl_agent,
        reward_list,
        reward_per_state_list,
        recent_level_rates,
        total_reward,
        step_num,
        completed_episodes,
    )

    if verbose and completed_episodes % 10 == 0:
        print("episode:%s level:%s success_rate:%s joint_success_rate:%s avg_reward:%s step_num:%s" % (
            completed_episodes,
            level_name,
            recent_success_rate,
            joint_success_rate,
            np.mean(reward_list[-50:]),
            step_num,
        ))
        print("level success:", recent_level_rates)

    if joint_success_rate >= best_joint_success_rate and completed_episodes > 100:
        best_joint_success_rate = joint_success_rate
        save_best_model(rl_agent, save_path, completed_episodes, best_joint_success_rate, recent_level_rates)

    if completed_episodes % 2000 == 0:
        rl_agent.save("%s/PPO_unpark_%s.pt" % (save_path, completed_episodes - 1), params_only=True)

    if verbose and completed_episodes % 20 == 0:
        save_reward_curve(save_path, reward_list)

    return best_joint_success_rate


def train_single_env(args, rl_agent, env, writer, save_path, level_chooser):
    reward_list = []
    reward_per_state_list = []
    succ_record = []
    best_joint_success_rate = 0.0

    for episode_idx in range(1, args.train_episode + 1):
        level_chosen = level_chooser.choose_level()
        obs = env.reset(None, None, level_chosen)
        done = False
        total_reward = 0
        step_num = 0

        while not done:
            step_num += 1
            action, log_prob = rl_agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            reward_per_state_list.append(reward)
            rl_agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            obs = next_obs
            maybe_update_agent(rl_agent, writer, episode_idx)

        success = int(info["status"] == Status.ARRIVED)
        best_joint_success_rate = handle_episode_end(
            rl_agent,
            writer,
            save_path,
            level_chooser,
            succ_record,
            reward_list,
            reward_per_state_list,
            total_reward,
            step_num,
            level_chosen,
            success,
            episode_idx,
            best_joint_success_rate,
            args.verbose,
        )


def train_parallel_env(args, rl_agent, parallel_env, writer, save_path, level_chooser):
    reward_list = []
    reward_per_state_list = []
    succ_record = []
    best_joint_success_rate = 0.0
    completed_episodes = 0

    current_levels = [level_chooser.choose_level() for _ in range(args.num_envs)]
    obs_batch = parallel_env.reset_all([(None, None, level) for level in current_levels])
    episode_rewards = [0.0 for _ in range(args.num_envs)]
    episode_steps = [0 for _ in range(args.num_envs)]

    while completed_episodes < args.train_episode:
        actions, log_probs = rl_agent.choose_actions(obs_batch)
        step_returns = parallel_env.step(actions)

        for env_idx, (next_obs, reward, done, info) in enumerate(step_returns):
            episode_rewards[env_idx] += reward
            episode_steps[env_idx] += 1
            reward_per_state_list.append(reward)
            rl_agent.push_memory((obs_batch[env_idx], actions[env_idx], reward, done, log_probs[env_idx], next_obs))
            obs_batch[env_idx] = next_obs

        maybe_update_agent(rl_agent, writer, completed_episodes)

        for env_idx, (_, _, done, info) in enumerate(step_returns):
            if not done:
                continue

            completed_episodes += 1
            success = int(info["status"] == Status.ARRIVED)
            best_joint_success_rate = handle_episode_end(
                rl_agent,
                writer,
                save_path,
                level_chooser,
                succ_record,
                reward_list,
                reward_per_state_list,
                episode_rewards[env_idx],
                episode_steps[env_idx],
                current_levels[env_idx],
                success,
                completed_episodes,
                best_joint_success_rate,
                args.verbose,
            )

            if completed_episodes >= args.train_episode:
                continue

            current_levels[env_idx] = level_chooser.choose_level()
            obs_batch[env_idx] = parallel_env.reset_one(env_idx, (None, None, current_levels[env_idx]))
            episode_rewards[env_idx] = 0.0
            episode_steps[env_idx] = 0


def run_evaluation(args, rl_agent, train_levels, save_path):
    eval_env = make_env(False, args.verbose, args.img_mode)
    eval_path = save_path + "/eval"
    os.makedirs(eval_path, exist_ok=True)
    eval_agent = ParkingAgent(rl_agent, None)
    with torch.no_grad():
        for level in train_levels:
            eval_env.set_level(level)
            level_eval_path = os.path.join(eval_path, level.lower())
            os.makedirs(level_eval_path, exist_ok=True)
            eval(eval_env, eval_agent, episode=args.eval_episode, log_path=level_eval_path, post_proc_action=True)
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_ckpt", type=str, default=None)
    parser.add_argument("--img_ckpt", type=str, default=os.path.join(ROOT_DIR, "model", "ckpt", "autoencoder.pt"))
    parser.add_argument("--train_episode", type=int, default=100000)
    parser.add_argument("--eval_episode", type=int, default=500)
    parser.add_argument("--levels", type=str, default="Complex,Extrem")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--start_method", type=str, default="spawn")
    parser.add_argument("--img_mode", type=str, default=UNPARK_IMG_MODE)
    parser.add_argument("--use_slot_channel", type=str2bool, default=UNPARK_USE_SLOT_CHANNEL)
    parser.add_argument("--verbose", type=str2bool, default=True)
    parser.add_argument("--visualize", type=str2bool, default=True)
    args = parser.parse_args()
    args.img_mode = resolve_unpark_img_mode(args.agent_ckpt, args.img_mode, args.use_slot_channel)
    args.use_slot_channel = args.img_mode == "rgb_slot"

    if args.visualize and args.num_envs > 1:
        raise ValueError("Parallel env rollout requires `--visualize False`")

    train_levels = parse_levels(args.levels)
    save_path = make_save_path()
    writer = SummaryWriter(save_path)
    copyfile(os.path.join(ROOT_DIR, "configs.py"), save_path + "/configs.txt")
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = None
    parallel_env = None
    try:
        if args.num_envs > 1:
            parallel_env = ParallelCarParkingOutEnv(
                args.num_envs,
                env_kwargs=build_env_kwargs(args.visualize, args.verbose, args.img_mode),
                base_seed=SEED,
                start_method=args.start_method,
            )
            observation_shape = parallel_env.observation_shape
            action_dim = parallel_env.action_dim
        else:
            env = make_env(args.visualize, args.verbose, args.img_mode)
            env.action_space.seed(SEED)
            observation_shape = env.observation_shape
            action_dim = env.action_space.shape[0]

        rl_agent = build_agent(observation_shape, action_dim, args)
        level_chooser = LevelChoose(train_levels)

        if args.num_envs > 1:
            train_parallel_env(args, rl_agent, parallel_env, writer, save_path, level_chooser)
        else:
            train_single_env(args, rl_agent, env, writer, save_path, level_chooser)

        run_evaluation(args, rl_agent, train_levels, save_path)
    finally:
        if env is not None:
            env.close()
        if parallel_env is not None:
            parallel_env.close()
        writer.close()
