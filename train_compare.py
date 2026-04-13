from __future__ import annotations

import argparse
import os
import time

import numpy as np

from src.config import EnvConfig, TrainConfig
from src.env import CooperativeGoalSearchEnv
from src.plotting import plot_robot_curves
from src.training import ExperimentRunner


def _print_run_banner(env_cfg: EnvConfig, train_cfg: TrainConfig, save_dir: str) -> None:
    print("=" * 60)
    print("Cooperative Goal Search / train_compare")
    print("=" * 60)
    print(f"  world:        {env_cfg.width:.0f}x{env_cfg.height:.0f} (Fig. 8 layout)")
    print(f"  grid:         {train_cfg.grid_x}x{train_cfg.grid_y}")
    print(f"  episodes:     {train_cfg.episodes}")
    print(f"  trials:       {train_cfg.trials}")
    print(f"  seed:         {train_cfg.seed}")
    print(f"  noise_std:    {env_cfg.noise_std}")
    print(f"  save_dir:     {save_dir}")
    print(f"  max_steps/ep: {env_cfg.max_steps_per_episode}")
    print("-" * 60)


def _summarize_best_paths(best_paths: dict) -> None:
    names = ["R1", "R2", "R3"]
    for name in names:
        pts = best_paths.get(name) or []
        if len(pts) > 1:
            steps = len(pts) - 1
            print(f"  {name}: best path {steps} steps ({len(pts)} waypoints)")
        else:
            print(f"  {name}: no successful episode path (export will use greedy fallback)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-aligned cooperative goal search experiment runner")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--verbose", action="store_true", help="Per-trial goal rate and step stats during training")
    args = parser.parse_args()

    t0 = time.perf_counter()
    env_cfg = EnvConfig()
    train_cfg = TrainConfig(episodes=args.episodes, trials=args.trials, save_dir=args.save_dir)
    runner = ExperimentRunner(env_cfg, train_cfg, verbose=args.verbose)

    _print_run_banner(env_cfg, train_cfg, args.save_dir)
    demo_env = CooperativeGoalSearchEnv(env_cfg, np.random.default_rng(0))
    print(demo_env.render_ascii_summary())
    print("-" * 60)

    result_files = {}
    n_methods = len(ExperimentRunner.METHODS)
    for i, method in enumerate(ExperimentRunner.METHODS):
        print(f"[{i + 1}/{n_methods}] Running method: {method} ...")
        t_m = time.perf_counter()
        result = runner.run_method(method)
        out_json = runner.save_result(method, result)
        dt = time.perf_counter() - t_m
        print(f"      done in {dt:.1f}s -> {out_json}")
        result_files[method] = out_json

    print("-" * 60)
    print("Plotting learning curves ...")
    t_p = time.perf_counter()
    plot_path = os.path.join(args.save_dir, "cooperative_goal_search_curves.png")
    plot_robot_curves(result_files, plot_path, smoothing_window=train_cfg.smoothing_window)
    print(f"  Saved plot ({time.perf_counter() - t_p:.1f}s): {plot_path}")

    print("-" * 60)
    print("Webots export (proposed, best successful paths per robot) ...")
    t_w = time.perf_counter()
    demo_env, demo_agents, best_paths = runner.train_for_webots_export(method="proposed")
    _summarize_best_paths(best_paths)
    path_file = os.path.join(args.save_dir, "webots_paths.json")
    runner.export_webots_paths(demo_env, demo_agents, path_file, best_paths=best_paths)
    print(f"  Saved ({time.perf_counter() - t_w:.1f}s): {path_file}")

    total = time.perf_counter() - t0
    print("=" * 60)
    print(f"Total wall time: {total:.1f}s ({total/60.0:.2f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
