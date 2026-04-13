from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .config import EnvConfig

Rect = Tuple[float, float, float, float]


@dataclass
class StepResult:
    position: np.ndarray
    reward: float
    done: bool
    collided: bool
    reached_goal: bool


def _wall_rect(center_x: float, y0: float, y1: float, thickness: float) -> Rect:
    h = thickness * 0.5
    return (center_x - h, y0, center_x + h, y1)


class CooperativeGoalSearchEnv:
    """300×300 grid: discrete actions, step length `step_size`, optional Gaussian noise."""

    ACTIONS = {
        0: np.array([0.0, 1.0]),   # up (+y)
        1: np.array([0.0, -1.0]),  # down
        2: np.array([-1.0, 0.0]),  # left
        3: np.array([1.0, 0.0]),   # right
    }

    def __init__(self, cfg: EnvConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.obstacles = self._build_obstacles()

    def _build_obstacles(self) -> List[Rect]:
        """Fig. 8: (1|2) opening only at bottom; (2|3) opening only in middle band."""
        t = self.cfg.wall_thickness
        xl = self.cfg.wall_x_left
        xr = self.cfg.wall_x_right
        h = self.cfg.height

        # Area 1 | Area 2: single passage at bottom y in [0, left_wall_y_start)
        left_upper = _wall_rect(xl, self.cfg.left_wall_y_start, h, t)

        gy0 = self.cfg.right_wall_gap_y0
        gy1 = self.cfg.right_wall_gap_y1
        # Area 2 | Area 3: block below gap, block above gap; passage y in (gy0, gy1)
        right_lo = _wall_rect(xr, 0.0, gy0, t)
        right_hi = _wall_rect(xr, gy1, h, t)

        mid = (135.0, 160.0, 165.0, 190.0)
        return [left_upper, right_lo, right_hi, mid]

    @staticmethod
    def _in_rect(point: np.ndarray, rect: Rect) -> bool:
        x, y = float(point[0]), float(point[1])
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def sample_start(self, robot_idx: int) -> np.ndarray:
        pts = [self.cfg.start_r1, self.cfg.start_r2, self.cfg.start_r3]
        x, y = pts[robot_idx]
        return np.array([float(x), float(y)], dtype=float)

    def reset_agent(self, robot_idx: int) -> np.ndarray:
        return self.sample_start(robot_idx)

    def is_goal(self, point: np.ndarray) -> bool:
        return self._in_rect(point, self.cfg.goal_rect)

    def is_collision(self, point: np.ndarray) -> bool:
        x, y = float(point[0]), float(point[1])
        w, h = self.cfg.width, self.cfg.height
        if x < 0.0 or x > w or y < 0.0 or y > h:
            return True
        return any(self._in_rect(point, rect) for rect in self.obstacles)

    def step(self, position: np.ndarray, action: int) -> StepResult:
        direction = self.ACTIONS[action]
        if self.cfg.noise_std > 0.0:
            noise = self.rng.normal(0.0, self.cfg.noise_std)
        else:
            noise = 0.0
        dist = self.cfg.step_size + noise
        proposed = position + direction * dist

        if self.is_collision(proposed):
            return StepResult(
                position.copy(),
                self.cfg.collision_reward,
                False,
                True,
                False,
            )

        if self.is_goal(proposed):
            return StepResult(proposed, self.cfg.goal_reward, True, False, True)

        return StepResult(proposed, self.cfg.step_reward, False, False, False)

    def greedy_rollout(
        self,
        policy_fn,
        robot_idx: int,
        max_steps: int,
        noise_std: float | None = None,
        start_pos: np.ndarray | None = None,
    ) -> Tuple[List[List[float]], bool, int]:
        """Greedy rollout; returns (path, reached_goal, steps_used)."""
        original_noise = self.cfg.noise_std
        if noise_std is not None:
            self.cfg.noise_std = noise_std
        try:
            pos = np.array(start_pos, dtype=float) if start_pos is not None else self.reset_agent(robot_idx)
            path: List[List[float]] = [[float(pos[0]), float(pos[1])]]
            reached = False
            steps_used = max_steps
            for step in range(max_steps):
                action = int(policy_fn(pos))
                result = self.step(pos, action)
                pos = result.position
                path.append([float(pos[0]), float(pos[1])])
                if result.reached_goal:
                    reached = True
                    steps_used = step + 1
                    break
            return path, reached, steps_used
        finally:
            self.cfg.noise_std = original_noise

    def render_ascii_summary(self) -> str:
        return (
            "World 300x300, Fig. 8 (bottom gap 1|2, middle gap 2|3).\n"
            f"Goal: {self.cfg.goal_rect}\n"
            f"Starts: {self.cfg.start_r1}, {self.cfg.start_r2}, {self.cfg.start_r3}\n"
            f"Obstacles: {self.obstacles}"
        )
