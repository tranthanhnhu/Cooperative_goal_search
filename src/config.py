from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvConfig:
    """300×300 world aligned with paper Fig. 8 (user-specified wall layout)."""

    width: float = 300.0
    height: float = 300.0
    step_size: float = 5.0
    # Paper: Gaussian noise σ=0.5 on step length. Webots replay uses best real trajectories.
    noise_std: float = 0.5

    # Goal ~ (150, 275), 10×10 region
    goal_rect: Tuple[float, float, float, float] = (145.0, 270.0, 155.0, 280.0)

    # Fixed starts (Python coords; same mapping as Webots export)
    start_r1: Tuple[float, float] = (25.0, 25.0)
    start_r2: Tuple[float, float] = (150.0, 25.0)
    start_r3: Tuple[float, float] = (275.0, 25.0)

    collision_reward: float = -100.0
    goal_reward: float = 100.0
    step_reward: float = -1e-6
    max_steps_per_episode: int = 2000

    wall_thickness: float = 5.0
    # Fig. 8: vertical dividers at x=100 (Area1|2) and x=200 (Area2|3)
    wall_x_left: float = 100.0
    wall_x_right: float = 200.0
    # Only bottom passage between Area 1 and 2: wall from this y up to height (no top gap)
    left_wall_y_start: float = 40.0
    # Only one passage between Area 2 and 3 (middle/upper): gap (y0,y1), walls below and above
    right_wall_gap_y0: float = 130.0
    right_wall_gap_y1: float = 210.0


@dataclass
class TrainConfig:
    """Training settings (trials=40 matches paper)."""

    episodes: int = 500
    trials: int = 40
    seed: int = 42

    grid_x: int = 100
    grid_y: int = 100
    actions: int = 4

    alpha: float = 0.2
    gamma: float = 0.98
    epsilon_start: float = 0.25
    epsilon_end: float = 0.05

    planning_steps: int = 10
    planning_depth: int = 5
    replay_lambda: float = 0.25

    cluster_distance_threshold: float = 0.16
    visit_threshold: int = 3
    share_t_threshold: float = 2.2

    normalize_reward_min: float = -100.0
    normalize_reward_max: float = 100.0

    smoothing_window: int = 15
    save_dir: str = "results"


@dataclass
class WebotsConfig:
    max_speed: float = 6.28
    target_tolerance: float = 0.05
    waypoints_scale: float = 0.01
    arena_half: float = 1.5
