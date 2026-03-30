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

from analysis.export_vector_case_plots import (
    PALETTE,
    ensure_dir,
    generate_scene_map,
    map_to_dict,
    save_case_figure,
    save_panel,
    run_dual_case,
    run_single_case,
)


SCAN_SPECS = [
    {"level": "Complex", "case_id": 0, "name": "Complex Bay", "tag": "complex_bay"},
    {"level": "Complex", "case_id": 1, "name": "Complex Parallel", "tag": "complex_parallel"},
    {"level": "Extrem", "case_id": 0, "name": "Extreme", "tag": "extreme"},
]


def sign_of_speed(action):
    speed = float(action[1])
    if speed > 1e-6:
        return 1
    if speed < -1e-6:
        return -1
    return 0


def count_gear_shifts(segments):
    total = 0
    prev_sign = None
    for seg in segments:
        sign = sign_of_speed(seg["action"])
        if prev_sign is not None and sign != 0 and prev_sign != 0 and sign != prev_sign:
            total += 1
        if sign != 0:
            prev_sign = sign
    return total


def summarize_case(level, case_id, seed, forward_ckpt, unpark_ckpt):
    map_obj = generate_scene_map(level, seed, case_id)
    action_seed = 41000 + case_id * 1000 + seed
    single = run_single_case(map_obj, forward_ckpt, action_seed)
    dual = run_dual_case(map_obj, forward_ckpt, unpark_ckpt, action_seed)
    single_gears = count_gear_shifts(single["segments"])
    dual_gears = count_gear_shifts(dual["segments"])
    return {
        "level": level,
        "case_id": case_id,
        "seed": seed,
        "map": map_to_dict(map_obj),
        "single": single,
        "dual": dual,
        "single_gear_shifts": single_gears,
        "dual_gear_shifts": dual_gears,
        "step_gain": int(single["step_num"] - dual["step_num"]),
        "rescued": bool((not single["final_success"]) and dual["final_success"]),
        "dual_better": bool(dual["final_success"] and ((not single["final_success"]) or dual["step_num"] < single["step_num"])),
    }


def candidate_score(row):
    rescue_bonus = 200 if row["rescued"] else 0
    success_bonus = 40 if row["dual"]["final_success"] else -200
    gain_bonus = max(row["step_gain"], 0)
    return rescue_bonus + success_bonus + gain_bonus - 15 * row["dual_gear_shifts"] - 0.6 * row["dual"]["step_num"]


def scan_spec(spec, max_seed, forward_ckpt, unpark_ckpt, min_step_gain, max_dual_gears, max_dual_steps):
    rows = []
    for seed in range(1, max_seed + 1):
        row = summarize_case(spec["level"], spec["case_id"], seed, forward_ckpt, unpark_ckpt)
        dual_ok = row["dual"]["final_success"] and row["dual_gear_shifts"] <= max_dual_gears and row["dual"]["step_num"] <= max_dual_steps
        improved = row["rescued"] or row["step_gain"] >= min_step_gain
        if dual_ok and improved:
            row["score"] = candidate_score(row)
            rows.append(row)
    rows.sort(
        key=lambda row: (
            row["dual_gear_shifts"],
            row["dual"]["step_num"],
            -row["step_gain"],
            -row["score"],
            row["seed"],
        )
    )
    return rows


def export_cases(selected_rows, out_dir, png_dpi):
    panel_inputs = []
    index = []
    for idx, row in enumerate(selected_rows):
        spec_label = row["level"] if row["level"] == "Extrem" else f"{row['level']} {'Bay' if row['case_id'] == 0 else 'Parallel'}"
        label = f"{spec_label} seed {row['seed']}"
        case_dir = out_dir / f"{row['level'].lower()}_{row['case_id']}_{row['seed']}"
        ensure_dir(case_dir)

        scene_summary = {
            "label": label,
            "level": row["level"],
            "case_id": row["case_id"],
            "seed": row["seed"],
            "single_steps": row["single"]["step_num"],
            "dual_steps": row["dual"]["step_num"],
            "single_gear_shifts": row["single_gear_shifts"],
            "dual_gear_shifts": row["dual_gear_shifts"],
            "step_gain": row["step_gain"],
            "rescued": row["rescued"],
            "connection_used": row["dual"]["connection_used"],
            "connection_index": row["dual"]["connection_index"],
            "score": row["score"],
        }

        (case_dir / "summary.json").write_text(json.dumps(scene_summary, indent=2))
        (case_dir / "scene_map.json").write_text(json.dumps(row["map"], indent=2))
        (case_dir / "single_trace.json").write_text(json.dumps(row["single"], indent=2))
        (case_dir / "dual_trace.json").write_text(json.dumps(row["dual"], indent=2))

        save_case_figure(row["single"], label, case_dir / "single_vector", png_dpi=png_dpi)
        save_case_figure(row["dual"], label, case_dir / "dual_vector", png_dpi=png_dpi)

        panel_inputs.append({"label": label, "single": row["single"], "dual": row["dual"]})
        index.append(scene_summary)

    if panel_inputs:
        save_panel(panel_inputs, out_dir, png_dpi=png_dpi)
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))


def write_notes(out_dir, selected_rows):
    lines = ["# Low-shift dual-model case gallery", ""]
    for row in selected_rows:
        spec_label = row["level"] if row["level"] == "Extrem" else f"{row['level']} {'Bay' if row['case_id'] == 0 else 'Parallel'}"
        lines.append(
            f"- {spec_label} seed {row['seed']}: dual steps {row['dual']['step_num']}, dual gear shifts {row['dual_gear_shifts']}, "
            f"single steps {row['single']['step_num']}, single gear shifts {row['single_gear_shifts']}, rescued={row['rescued']}."
        )
    (out_dir / "notes.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward_ckpt", type=str, default="src/model/ckpt/HOPE_SAC0.pt")
    parser.add_argument("--unpark_ckpt", type=str, default="src/log/exp/unpark_ppo_20260323_120610/PPO_unpark_best.pt")
    parser.add_argument("--max_seed", type=int, default=120)
    parser.add_argument("--per_spec", type=int, default=2)
    parser.add_argument("--min_step_gain", type=int, default=15)
    parser.add_argument("--max_dual_gears", type=int, default=3)
    parser.add_argument("--max_dual_steps", type=int, default=35)
    parser.add_argument("--png_dpi", type=int, default=480)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = Path(args.save_dir) if args.save_dir else Path(ROOT_DIR) / "log" / "analysis" / f"low_shift_gallery_{timestamp}"
    ensure_dir(save_dir)

    all_selected = []
    scan_manifest = {}

    for spec_idx, spec in enumerate(SCAN_SPECS):
        print(f"[{spec_idx + 1}/{len(SCAN_SPECS)}] scan {spec['name']}")
        rows = scan_spec(
            spec,
            max_seed=args.max_seed,
            forward_ckpt=args.forward_ckpt,
            unpark_ckpt=args.unpark_ckpt,
            min_step_gain=args.min_step_gain,
            max_dual_gears=args.max_dual_gears,
            max_dual_steps=args.max_dual_steps,
        )
        scan_manifest[spec["tag"]] = [
            {
                "seed": row["seed"],
                "single_steps": row["single"]["step_num"],
                "dual_steps": row["dual"]["step_num"],
                "single_gear_shifts": row["single_gear_shifts"],
                "dual_gear_shifts": row["dual_gear_shifts"],
                "step_gain": row["step_gain"],
                "rescued": row["rescued"],
                "score": row["score"],
            }
            for row in rows
        ]
        all_selected.extend(rows[: args.per_spec])

    (save_dir / "scan_manifest.json").write_text(json.dumps(scan_manifest, indent=2))

    all_selected.sort(key=lambda row: (row["dual_gear_shifts"], row["dual"]["step_num"], -row["step_gain"], row["seed"]))
    export_cases(all_selected, save_dir, png_dpi=args.png_dpi)
    write_notes(save_dir, all_selected)

    summary = {
        "save_dir": str(save_dir),
        "num_cases": len(all_selected),
        "cases": [
            {
                "level": row["level"],
                "case_id": row["case_id"],
                "seed": row["seed"],
                "single_steps": row["single"]["step_num"],
                "dual_steps": row["dual"]["step_num"],
                "single_gear_shifts": row["single_gear_shifts"],
                "dual_gear_shifts": row["dual_gear_shifts"],
                "step_gain": row["step_gain"],
                "rescued": row["rescued"],
            }
            for row in all_selected
        ],
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
