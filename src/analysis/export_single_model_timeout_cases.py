import argparse
import json
import os
import sys
import time
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

from export_failure_efficiency_examples import (
    PALETTE,
    build_sac_agent,
    create_pure_rl_env,
    draw_rl_trajectory,
    generate_scene_map,
    run_pure_rl_case,
    save_figure,
)


SELECTED_TIMEOUT_CASES = [
    {"key": "complex_bay_seed22", "label": "Complex Bay", "level": "Complex", "map_case_id": 0, "scene_seed": 22},
    {"key": "complex_bay_seed55", "label": "Complex Bay", "level": "Complex", "map_case_id": 0, "scene_seed": 55},
    {"key": "complex_parallel_seed9", "label": "Complex Parallel", "level": "Complex", "map_case_id": 1, "scene_seed": 9},
    {"key": "complex_parallel_seed13", "label": "Complex Parallel", "level": "Complex", "map_case_id": 1, "scene_seed": 13},
    {"key": "extrem_parallel_seed6263", "label": "Extrem Parallel", "level": "Extrem", "map_case_id": 1, "scene_seed": 6263},
    {"key": "extrem_parallel_seed41", "label": "Extrem Parallel", "level": "Extrem", "map_case_id": 1, "scene_seed": 41},
]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def render_single_case(result, case, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    draw_rl_trajectory(ax, result, title=f"{case['label']} single model timeout")
    save_figure(fig, output_dir / "figures" / f"{case['key']}_single_model_timeout", dpi)


def render_overview(results, output_dir, dpi):
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 15.3), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, item in zip(axes, results):
        draw_rl_trajectory(
            ax,
            item["result"],
            title=f"{item['case']['label']} seed {item['case']['scene_seed']} | single model timeout",
        )
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle("Single Model Timeout Failure Cases", fontsize=20, color=PALETTE["dark"], fontweight="semibold")
    save_figure(fig, output_dir / "overview_single_model_timeout_cases", dpi)


def parse_args():
    parser = argparse.ArgumentParser(description="Export selected single-model timeout failure cases.")
    parser.add_argument("--forward-ckpt", default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--output-dir", default="src/log/paper_support/single_model_timeout_cases_hidpi")
    parser.add_argument("--action-seed-base", type=int, default=91000)
    parser.add_argument("--rl-action-mode", choices=["choose", "raw"], default="choose")
    parser.add_argument("--png-dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = create_pure_rl_env()
    agent = build_sac_agent(args.forward_ckpt, env)
    env.close()

    items = []
    manifest_cases = []
    for case in SELECTED_TIMEOUT_CASES:
        map_obj = generate_scene_map(case["level"], case["scene_seed"], case["map_case_id"])
        action_seed = int(args.action_seed_base + case["scene_seed"] + 10000 * case["map_case_id"])
        result = run_pure_rl_case(map_obj, agent, action_seed, args.rl_action_mode)
        if result["status"] != "OUTTIME":
            print(f"[warn] {case['key']} status={result['status']} steps={result['step_num']}")
        render_single_case(result, case, output_dir, args.png_dpi)
        write_json(output_dir / "case_json" / f"{case['key']}_single_model_timeout.json", {"case": case, "result": result})
        items.append({"case": case, "result": result})
        manifest_cases.append(
            {
                **case,
                "action_seed": action_seed,
                "status": result["status"],
                "steps": result["step_num"],
                "final_success": result["final_success"],
                "rs_assist_used": result["rs_assist_used"],
                "action_mode": result["action_mode"],
            }
        )
        print(f"[case] {case['key']} status={result['status']} steps={result['step_num']}")

    render_overview(items, output_dir, args.png_dpi)
    write_json(
        output_dir / "manifest.json",
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "forward_ckpt": str(Path(args.forward_ckpt).resolve()),
            "output_dir": str(output_dir),
            "definition": "Single forward SAC model with RS assist disabled; timeout means OUTTIME at 200 rollout steps.",
            "args": vars(args),
            "cases": manifest_cases,
        },
    )
    print(f"[done] output_dir={output_dir}")


if __name__ == "__main__":
    main()
