import argparse
import os
from typing import List, Optional, Tuple

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CURVE_SPECS = {
    "reward_curve": {
        "title": {"en": "Reward Curve", "zh": "奖励曲线"},
        "ylabel": {"en": "Reward", "zh": "奖励值"},
        "series": [
            ("total_reward", {"en": "Total Reward", "zh": "总奖励"}),
        ],
    },
    "step_curve": {
        "title": {"en": "Completion Step Curve", "zh": "完成步数曲线"},
        "ylabel": {"en": "Step Number", "zh": "完成步数"},
        "series": [
            ("step_num", {"en": "Step Number", "zh": "完成步数"}),
        ],
    },
    "success_rate_curve": {
        "title": {"en": "Success Rate Curve", "zh": "成功率曲线"},
        "ylabel": {"en": "Success Rate", "zh": "成功率"},
        "series": [
            ("success_rate_unpark", {"en": "Unpark", "zh": "泊出成功率"}),
            ("success_rate_joint", {"en": "Joint", "zh": "联合成功率"}),
            ("success_rate_Normal", {"en": "Normal", "zh": "Normal"}),
            ("success_rate_Complex", {"en": "Complex", "zh": "Complex"}),
            ("success_rate_Extrem", {"en": "Extrem", "zh": "Extrem"}),
            ("success_rate_dlp", {"en": "DLP", "zh": "DLP"}),
        ],
    },
}


class CurveSegment:
    def __init__(self, log_dir: str, offset: int = 0, start_step: int = 0, end_step: Optional[int] = None):
        self.log_dir = log_dir
        self.offset = int(offset)
        self.start_step = int(start_step)
        self.end_step = None if end_step is None else int(end_step)


def configure_matplotlib(lang: str):
    if lang == "zh":
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_scalar_series(log_dir: str, tag: str) -> Tuple[List[int], List[float]]:
    accumulator = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return [], []
    events = accumulator.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def latest_event_dir(log_dir: str) -> str:
    event_files = [
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.startswith("events.out.tfevents")
    ]
    if not event_files:
        raise FileNotFoundError("No TensorBoard event file found in %s" % log_dir)
    return log_dir


def parse_segment(spec: str) -> CurveSegment:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) > 4:
        raise ValueError("Invalid segment spec: %s" % spec)
    while len(parts) < 4:
        parts.append("")
    log_dir, offset, start_step, end_step = parts
    if not log_dir:
        raise ValueError("Missing log_dir in segment spec: %s" % spec)
    return CurveSegment(
        log_dir=log_dir,
        offset=int(offset or 0),
        start_step=int(start_step or 0),
        end_step=None if end_step == "" else int(end_step),
    )


def build_segments(log_dir: Optional[str], segment_specs: List[str]) -> List[CurveSegment]:
    if segment_specs:
        segments = [parse_segment(spec) for spec in segment_specs]
    elif log_dir is not None:
        segments = [CurveSegment(log_dir=log_dir)]
    else:
        raise ValueError("Either `--log_dir` or at least one `--segment` must be provided")

    for segment in segments:
        latest_event_dir(segment.log_dir)
    return segments


def collect_segment_series(segment: CurveSegment, tag: str) -> Tuple[List[int], List[float]]:
    xs, ys = load_scalar_series(segment.log_dir, tag)
    if not xs:
        return [], []

    filtered = []
    for x, y in zip(xs, ys):
        if x < segment.start_step:
            continue
        if segment.end_step is not None and x > segment.end_step:
            continue
        filtered.append((x + segment.offset, y))

    if not filtered:
        return [], []
    xs, ys = zip(*filtered)
    return list(xs), list(ys)


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 1:
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


def format_k_tick(value, _pos):
    scaled = value / 1000.0
    if abs(scaled - round(scaled)) < 1e-8:
        return str(int(round(scaled)))
    return "%.1f" % scaled


def plot_curves(
    segments: List[CurveSegment],
    output_dir: str,
    max_episode: int = None,
    min_episode: int = 0,
    lang: str = "en",
    smooth_window: int = 1,
    ymin_zero: bool = False,
    reward_origin_zero: bool = False,
    success_rate_origin_zero: bool = False,
    success_rate_origin_pre_smooth: bool = False,
    exclude_tags: Optional[List[str]] = None,
    success_rate_percent: bool = False,
    success_rate_ylim_min: Optional[float] = None,
    success_rate_ylim_max: Optional[float] = None,
    title_fontsize: int = 18,
    label_fontsize: int = 16,
    tick_fontsize: int = 14,
    legend_fontsize: int = 14,
    line_width: float = 2.0,
    x_axis_k: bool = False,
    x_axis_sci: bool = False,
    fig_width: float = 8.0,
    fig_height: float = 4.5,
    dpi: int = 150,
    hide_legend: bool = False,
    curve_color: Optional[str] = None,
    success_curve_color: Optional[str] = None,
    keep_x_origin_when_trimmed: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    configure_matplotlib(lang)
    exclude_tags = set(exclude_tags or [])

    xlabel = {"en": "Episode", "zh": "训练轮次"}[lang]
    if x_axis_k:
        xlabel = {"en": "Episode (k)", "zh": "训练轮次 (k)"}[lang]

    for stem, curve_spec in CURVE_SPECS.items():
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        plotted = False
        for tag, labels in curve_spec["series"]:
            if tag in exclude_tags:
                continue
            all_points = []
            for segment in segments:
                xs, ys = collect_segment_series(segment, tag)
                if not xs:
                    continue
                all_points.extend(zip(xs, ys))
            if not all_points:
                continue
            all_points.sort(key=lambda item: item[0])
            if max_episode is not None:
                all_points = [item for item in all_points if item[0] <= max_episode]
                if not all_points:
                    continue
            xs, ys = zip(*all_points)
            xs = list(xs)
            ys = list(ys)
            if stem == "success_rate_curve" and success_rate_origin_pre_smooth and xs and ys:
                if xs[0] != 0 or ys[0] != 0:
                    xs = [0] + xs
                    ys = [0.0] + ys
            ys = moving_average(list(ys), smooth_window)
            if min_episode is not None:
                filtered = [(x, y) for x, y in zip(xs, ys) if x >= min_episode]
                if not filtered:
                    continue
                xs, ys = zip(*filtered)
                xs = list(xs)
                ys = list(ys)
            if stem == "reward_curve" and reward_origin_zero and xs and ys and (min_episode is None or min_episode <= 0):
                if xs[0] != 0 or ys[0] != 0:
                    xs = [0] + xs
                    ys = [0.0] + ys
            if (
                stem == "success_rate_curve"
                and success_rate_origin_zero
                and xs
                and ys
                and (min_episode is None or min_episode <= 0)
            ):
                if xs[0] != 0 or ys[0] != 0:
                    xs = [0] + xs
                    ys = [0.0] + ys
            if stem == "success_rate_curve" and success_rate_percent:
                ys = [value * 100.0 for value in ys]
            plot_kwargs = {"label": labels[lang], "linewidth": line_width}
            chosen_color = curve_color
            if stem == "success_rate_curve" and success_curve_color:
                chosen_color = success_curve_color
            if chosen_color:
                plot_kwargs["color"] = chosen_color
            ax.plot(xs, ys, **plot_kwargs)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(curve_spec["ylabel"][lang], fontsize=label_fontsize)
        ax.set_title(curve_spec["title"][lang], fontsize=title_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        if max_episode is not None:
            if keep_x_origin_when_trimmed:
                x_min = 0
            else:
                x_min = 0 if min_episode is None else max(0, min_episode)
            ax.set_xlim(x_min, max_episode)
        if x_axis_sci:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(formatter)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_size(tick_fontsize)
        elif x_axis_k:
            ax.xaxis.set_major_formatter(FuncFormatter(format_k_tick))
        if ymin_zero:
            ax.set_ylim(bottom=0)
        if stem == "success_rate_curve":
            if success_rate_percent:
                ax.set_ylabel({"en": "Success Rate (%)", "zh": "成功率 (%)"}[lang], fontsize=label_fontsize)
            current_bottom, current_top = ax.get_ylim()
            bottom = success_rate_ylim_min if success_rate_ylim_min is not None else current_bottom
            top = success_rate_ylim_max if success_rate_ylim_max is not None else current_top
            ax.set_ylim(bottom=bottom, top=top)
        ax.grid(alpha=0.3)
        if not hide_legend:
            ax.legend(fontsize=legend_fontsize)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "%s.png" % stem))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="segment spec: log_dir,offset,start_step,end_step ; later segments can be stitched onto earlier ones",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_episode", type=int, default=100000)
    parser.add_argument("--min_episode", type=int, default=0)
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--ymin_zero", action="store_true")
    parser.add_argument("--reward_origin_zero", action="store_true")
    parser.add_argument("--success_rate_origin_zero", action="store_true")
    parser.add_argument("--success_rate_origin_pre_smooth", action="store_true")
    parser.add_argument("--exclude_tag", action="append", default=[])
    parser.add_argument("--success_rate_percent", action="store_true")
    parser.add_argument("--success_rate_ylim_min", type=float, default=None)
    parser.add_argument("--success_rate_ylim_max", type=float, default=None)
    parser.add_argument("--title_fontsize", type=int, default=18)
    parser.add_argument("--label_fontsize", type=int, default=16)
    parser.add_argument("--tick_fontsize", type=int, default=14)
    parser.add_argument("--legend_fontsize", type=int, default=14)
    parser.add_argument("--line_width", type=float, default=2.0)
    parser.add_argument("--x_axis_k", action="store_true")
    parser.add_argument("--x_axis_sci", action="store_true")
    parser.add_argument("--fig_width", type=float, default=8.0)
    parser.add_argument("--fig_height", type=float, default=4.5)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--hide_legend", action="store_true")
    parser.add_argument("--curve_color", type=str, default=None)
    parser.add_argument("--success_curve_color", type=str, default=None)
    parser.add_argument("--keep_x_origin_when_trimmed", action="store_true")
    args = parser.parse_args()

    segments = build_segments(args.log_dir, args.segment)
    output_dir = args.output_dir or segments[0].log_dir
    plot_curves(
        segments,
        output_dir,
        max_episode=args.max_episode,
        min_episode=args.min_episode,
        lang=args.lang,
        smooth_window=args.smooth_window,
        ymin_zero=args.ymin_zero,
        reward_origin_zero=args.reward_origin_zero,
        success_rate_origin_zero=args.success_rate_origin_zero,
        success_rate_origin_pre_smooth=args.success_rate_origin_pre_smooth,
        exclude_tags=args.exclude_tag,
        success_rate_percent=args.success_rate_percent,
        success_rate_ylim_min=args.success_rate_ylim_min,
        success_rate_ylim_max=args.success_rate_ylim_max,
        title_fontsize=args.title_fontsize,
        label_fontsize=args.label_fontsize,
        tick_fontsize=args.tick_fontsize,
        legend_fontsize=args.legend_fontsize,
        line_width=args.line_width,
        x_axis_k=args.x_axis_k,
        x_axis_sci=args.x_axis_sci,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        dpi=args.dpi,
        hide_legend=args.hide_legend,
        curve_color=args.curve_color,
        success_curve_color=args.success_curve_color,
        keep_x_origin_when_trimmed=args.keep_x_origin_when_trimmed,
    )


if __name__ == "__main__":
    main()
