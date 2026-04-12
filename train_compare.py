from __future__ import annotations

import argparse
import os

from src.config import EnvConfig, TrainConfig
from src.plotting import plot_robot_curves
from src.training import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-aligned cooperative goal search experiment runner")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--verbose", action="store_true", help="Print per-trial goal rates during training")
    args = parser.parse_args()

    env_cfg = EnvConfig()
    train_cfg = TrainConfig(episodes=args.episodes, trials=args.trials, save_dir=args.save_dir)
    runner = ExperimentRunner(env_cfg, train_cfg, verbose=args.verbose)

    result_files = {}
    for method in ExperimentRunner.METHODS:
        print(f"Running method: {method}")
        result = runner.run_method(method)
        result_files[method] = runner.save_result(method, result)

    plot_path = os.path.join(args.save_dir, "cooperative_goal_search_curves.png")
    plot_robot_curves(result_files, plot_path, smoothing_window=train_cfg.smoothing_window)
    print(f"Saved comparison plot to: {plot_path}")

    print("Training proposed model for Webots (best successful paths)...")
    demo_env, demo_agents, best_paths = runner.train_for_webots_export(method="proposed")
    path_file = os.path.join(args.save_dir, "webots_paths.json")
    runner.export_webots_paths(demo_env, demo_agents, path_file, best_paths=best_paths)
    print(f"Saved Webots replay paths to: {path_file}")


if __name__ == "__main__":
    main()
