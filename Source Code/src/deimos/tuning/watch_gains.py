"""
Print the current best-known Kp/Kd gains from a checkpoint file, while the
GA is (or isn't) still running. Read-only -- safe to run alongside the
actual GA process at any time, as many times as you like.

Usage:
    python watch_gains.py results\ga_checkpoint_PD_seed0_p50g80.pkl
    python watch_gains.py results\ga_checkpoint_PD_seed0_p50g80.pkl --generation 20
    python watch_gains.py results\ga_checkpoint_PD_seed0_p50g80.pkl --watch   # auto-refresh
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from deimos.tuning.objectives import history_gains, OBJECTIVE_LABELS, pareto_picks


def _load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None


def show(path, controller_type, generation):
    data = _load(path)
    if data is None:
        print(f"(no checkpoint at {path} yet)")
        return False

    history = data["history"]
    gen_idx = generation if generation is not None else -1
    entry = history[gen_idx]
    labels = OBJECTIVE_LABELS.get(controller_type, None)

    print(f"\n=== generation {entry['generation']}  "
          f"(front size {entry['front_size']}, {len(history)} total recorded) ===")

    # front-only decode always available, even without store_full_population
    gains = history_gains(history, controller_type=controller_type,
                           generation=gen_idx, front_only=True)

    picks = pareto_picks(gains["objectives"], controller_type)
    for label, idx in picks.items():
        obj_str = (", ".join(f"{lab}={v:.5g}" for lab, v in zip(labels, gains["objectives"][idx]))
                   if labels else str(gains["objectives"][idx]))
        print(f"  [{label:16s}] Kp={np.round(gains['Kp'][idx], 6)}  "
              f"Kd={np.round(gains['Kd'][idx], 6)}  ({obj_str})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_path")
    ap.add_argument("--controller-type", default="PD", choices=["PD", "PID"])
    ap.add_argument("--generation", type=int, default=None,
                     help="index into history; default -1 (latest recorded)")
    ap.add_argument("--watch", action="store_true",
                     help="keep polling and reprinting every --interval seconds")
    ap.add_argument("--interval", type=float, default=15.0)
    args = ap.parse_args()

    path = Path(args.checkpoint_path)

    if not args.watch:
        show(path, args.controller_type, args.generation)
        return

    print(f"watching {path} every {args.interval}s (Ctrl+C to stop)")
    last_gen = None
    try:
        while True:
            data = _load(path)
            if data is not None:
                latest_gen = data["history"][-1]["generation"]
                if latest_gen != last_gen:
                    show(path, args.controller_type, args.generation)
                    last_gen = latest_gen
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped (GA run unaffected)")


if __name__ == "__main__":
    main()
