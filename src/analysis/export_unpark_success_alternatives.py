import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def configure_matplotlib():
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_scalar(log_dir, tag):
    accumulator = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        raise ValueError("TensorBoard tag `%s` not found in %s" % (tag, log_dir))
    return [(event.step, float(event.value)) for event in accumulator.Scalars(tag)]


def reconstruct_binary_successes(rate_points, window):
    successes = []
    for step, rate in rate_points:
        if step <= window:
            cumulative_success = int(round(rate * step))
            previous_success = sum(successes)
            success = cumulative_success - previous_success
        else:
            window_success = int(round(rate * window))
            previous_window_success = sum(successes[-(window - 1):])
            success = window_success - previous_window_success
        successes.append(int(min(max(success, 0), 1)))
    return successes


def moving_average(values, window):
    if window <= 1:
        return list(values)
    smoothed = []
    running_sum = 0.0
    history = []
    for value in values:
        history.append(value)
        running_sum += value
        if len(history) > window:
            running_sum -= history.pop(0)
        smoothed.append(running_sum / len(history))
    return smoothed


def export_fixed_denominator_curve(log_dir, output_dir, window=100, max_episode=None, smooth_window=1):
    points = load_scalar(log_dir, "success_rate_unpark")
    if max_episode is not None:
        points = [(step, rate) for step, rate in points if step <= max_episode]
    successes = reconstruct_binary_successes(points, window)

    xs = [0]
    ys = [0.0]
    for idx, (step, _rate) in enumerate(points):
        start = max(0, idx - window + 1)
        fixed_success_rate = sum(successes[start:idx + 1]) / float(window)
        xs.append(step)
        ys.append(fixed_success_rate * 100.0)
    plot_ys = moving_average(ys, smooth_window)

    os.makedirs(output_dir, exist_ok=True)
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=240)
    label = "固定分母100成功率" if smooth_window <= 1 else "固定分母100成功率（平滑%s轮）" % smooth_window
    ax.plot(xs, plot_ys, linewidth=2.2, label=label)
    ax.set_xlim(0, max(xs))
    ax.set_ylim(0, 100)
    ax.set_xlabel("训练轮次", fontsize=16)
    ax.set_ylabel("成功率 (%)", fontsize=16)
    ax.set_title("固定分母成功率曲线", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "success_rate_fixed100_curve.png"))
    if smooth_window > 1:
        fig.savefig(os.path.join(output_dir, "success_rate_fixed100_smooth%s_curve.png" % smooth_window))
    plt.close(fig)


def read_fixed_eval_csv(csv_path):
    grouped = defaultdict(lambda: defaultdict(dict))
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode = int(row["episode"])
            level = row["level"]
            grouped[episode][level] = {
                "success_rate": float(row["success_rate"]),
                "avg_reward": float(row["avg_reward"]),
                "avg_step": float(row["avg_step"]),
            }
    return grouped


def export_fixed_eval_curves(csv_path, output_dir):
    grouped = read_fixed_eval_csv(csv_path)
    if not grouped:
        raise ValueError("No rows found in %s" % csv_path)

    episodes = sorted(grouped.keys())
    levels = sorted({level for row in grouped.values() for level in row.keys()})
    os.makedirs(output_dir, exist_ok=True)
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=240)
    for level in levels:
        ys = [grouped[episode][level]["success_rate"] * 100.0 for episode in episodes]
        ax.plot(episodes, ys, linewidth=2.0, marker="o", markersize=3, label=level)
    joint = [min(grouped[episode][level]["success_rate"] for level in levels) * 100.0 for episode in episodes]
    ax.plot(episodes, joint, linewidth=2.4, marker="s", markersize=3, label="Joint")
    ax.set_xlim(0, max(episodes))
    ax.set_ylim(0, 100)
    ax.set_xlabel("训练轮次", fontsize=16)
    ax.set_ylabel("固定测试集成功率 (%)", fontsize=16)
    ax.set_title("固定测试集成功率曲线", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fixed_eval_success_curve.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=240)
    for level in levels:
        ys = [grouped[episode][level]["avg_reward"] for episode in episodes]
        ax.plot(episodes, ys, linewidth=2.0, marker="o", markersize=3, label=level)
    mean_rewards = [
        sum(grouped[episode][level]["avg_reward"] for level in levels) / max(len(levels), 1)
        for episode in episodes
    ]
    ax.plot(episodes, mean_rewards, linewidth=2.6, marker="s", markersize=3, label="Mean")
    ax.set_xlim(0, max(episodes))
    ax.set_xlabel("训练轮次", fontsize=16)
    ax.set_ylabel("平均奖励", fontsize=16)
    ax.set_title("固定测试集奖励曲线", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fixed_eval_reward_curve.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=240)
    ax.plot(episodes, mean_rewards, linewidth=2.6, marker="o", markersize=3, color="#2A9D8F")
    ax.set_xlim(0, max(episodes))
    ax.set_xlabel("训练轮次", fontsize=16)
    ax.set_ylabel("综合平均奖励", fontsize=16)
    ax.set_title("综合平均奖励曲线", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fixed_eval_mean_reward_curve.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--max_episode", type=int, default=None)
    parser.add_argument("--fixed_eval_csv", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.log_dir, "alt_success_curves")
    export_fixed_denominator_curve(args.log_dir, output_dir, args.window, args.max_episode, args.smooth_window)
    if args.fixed_eval_csv:
        export_fixed_eval_curves(args.fixed_eval_csv, output_dir)


if __name__ == "__main__":
    main()
