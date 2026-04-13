from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy
import json
import os
import numpy as np

from .agent import CooperativeAgent
from .config import EnvConfig, TrainConfig
from .env import CooperativeGoalSearchEnv


@dataclass
class TrialResult:
    steps_per_agent: Dict[str, List[float]]


def _pt_close(p: List[float], q: List[float], eps: float = 1e-2) -> bool:
    return abs(p[0] - q[0]) < eps and abs(p[1] - q[1]) < eps


def _trim_trailing_oscillation(path: List[List[float]]) -> List[List[float]]:
    """Remove trailing A-B-A-B oscillation from greedy fallback."""
    out = copy.deepcopy(path)
    while len(out) >= 4:
        a, b, c, d = out[-4], out[-3], out[-2], out[-1]
        if _pt_close(a, c) and _pt_close(b, d) and not _pt_close(a, b):
            out = out[:-2]
        else:
            break
    return out


class ExperimentRunner:
    """Runs paper-style comparison methods in the fast Python simulator."""

    METHODS = ["dyna_no_sharing", "raw_sharing", "request_sharing", "proposed"]

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig, verbose: bool = False):
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.verbose = verbose
        os.makedirs(train_cfg.save_dir, exist_ok=True)

    def _make_rng(self, seed_offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.train_cfg.seed + seed_offset)

    def _make_agents(self, rng: np.random.Generator) -> List[CooperativeAgent]:
        return [
            CooperativeAgent(i, self.train_cfg, self.env_cfg, np.random.default_rng(rng.integers(1_000_000_000)))
            for i in range(3)
        ]

    def _epsilon_for_episode(self, episode: int) -> float:
        frac = episode / max(self.train_cfg.episodes - 1, 1)
        return self.train_cfg.epsilon_start + frac * (self.train_cfg.epsilon_end - self.train_cfg.epsilon_start)

    @staticmethod
    def _run_episode(
        env: CooperativeGoalSearchEnv,
        agent: CooperativeAgent,
        robot_idx: int,
        epsilon: float,
        agents: List[CooperativeAgent],
        method: str,
        record_path: bool,
    ) -> Tuple[int, bool, List[List[float]]]:
        pos = env.reset_agent(robot_idx)
        trajectory: List[List[float]] = [[float(pos[0]), float(pos[1])]]
        steps_taken = env.cfg.max_steps_per_episode
        reached = False

        for step in range(env.cfg.max_steps_per_episode):
            state = agent.discretize(pos)
            action = agent.select_action_epsilon_greedy(state, epsilon)
            result = env.step(pos, action)
            next_state = agent.discretize(result.position)
            agent.update_from_real_experience(state, action, result.reward, next_state, pos, result.position)

            # Tan-style sharing: teammates receive Q-updates only (no duplicate model clusters).
            if method == "raw_sharing":
                for teammate in agents:
                    if teammate.agent_id != agent.agent_id:
                        teammate.direct_q_update(state, action, result.reward, next_state)

            pos = result.position
            if record_path:
                trajectory.append([float(pos[0]), float(pos[1])])

            if result.reached_goal:
                steps_taken = step + 1
                reached = True
                break

        return steps_taken, reached, trajectory

    def run_method(self, method: str) -> TrialResult:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}")

        per_trial_histories = {"robot_1": [], "robot_2": [], "robot_3": []}

        for trial in range(self.train_cfg.trials):
            rng = self._make_rng(seed_offset=trial * 10000)
            env = CooperativeGoalSearchEnv(self.env_cfg, rng)
            agents = self._make_agents(rng)
            histories = {"robot_1": [], "robot_2": [], "robot_3": []}
            successes = {"robot_1": 0, "robot_2": 0, "robot_3": 0}
            steps_on_success: Dict[str, List[float]] = {"robot_1": [], "robot_2": [], "robot_3": []}
            no_goal = {"robot_1": 0, "robot_2": 0, "robot_3": 0}

            for episode in range(self.train_cfg.episodes):
                epsilon = self._epsilon_for_episode(episode)
                for idx, agent in enumerate(agents):
                    steps_taken, reached, _ = self._run_episode(
                        env, agent, idx, epsilon, agents, method, record_path=False
                    )
                    agent.planning_replay(agents, method)
                    key = f"robot_{idx + 1}"
                    histories[key].append(float(steps_taken))
                    if reached:
                        successes[key] += 1
                        steps_on_success[key].append(float(steps_taken))
                    else:
                        no_goal[key] += 1

            for robot_key in histories:
                per_trial_histories[robot_key].append(histories[robot_key])

            if self.verbose:
                ep = self.train_cfg.episodes
                parts = [
                    f"  [{method}] trial {trial + 1}/{self.train_cfg.trials}",
                    f"goal R1={successes['robot_1']/ep:.3f} R2={successes['robot_2']/ep:.3f} R3={successes['robot_3']/ep:.3f}",
                ]
                for rk in ("robot_1", "robot_2", "robot_3"):
                    sos = steps_on_success[rk]
                    ms = float(np.mean(sos)) if sos else float("nan")
                    parts.append(f"{rk[-1]}_steps_ok={ms:.1f}" if sos else f"{rk[-1]}_steps_ok=nan")
                    parts.append(f"{rk[-1]}_no_goal={no_goal[rk]/ep:.3f}")
                print(" | ".join(parts))

        averaged = {
            robot_key: np.mean(np.array(trials_hist, dtype=float), axis=0).tolist()
            for robot_key, trials_hist in per_trial_histories.items()
        }
        return TrialResult(steps_per_agent=averaged)

    def train_for_webots_export(
        self,
        method: str = "proposed",
    ) -> Tuple[CooperativeGoalSearchEnv, List[CooperativeAgent], Dict[str, List[List[float]]]]:
        """Train agents and track best successful trajectory per robot for Webots."""
        rng = self._make_rng(seed_offset=777777)
        env = CooperativeGoalSearchEnv(self.env_cfg, rng)
        agents = self._make_agents(rng)

        best_steps: Dict[int, float] = {0: float("inf"), 1: float("inf"), 2: float("inf")}
        best_paths: Dict[int, List[List[float]]] = {0: [], 1: [], 2: []}

        for episode in range(self.train_cfg.episodes):
            epsilon = self._epsilon_for_episode(episode)
            for idx, agent in enumerate(agents):
                steps_taken, reached, trajectory = self._run_episode(
                    env, agent, idx, epsilon, agents, method, record_path=True
                )
                if reached and steps_taken < best_steps[idx]:
                    best_steps[idx] = float(steps_taken)
                    best_paths[idx] = copy.deepcopy(trajectory)

                agent.planning_replay(agents, method)

        names = ["R1", "R2", "R3"]
        out: Dict[str, List[List[float]]] = {names[i]: best_paths[i] for i in range(3)}
        return env, agents, out

    def export_webots_paths(
        self,
        env: CooperativeGoalSearchEnv,
        agents: List[CooperativeAgent],
        output_path: str,
        best_paths: Optional[Dict[str, List[List[float]]]] = None,
    ) -> str:
        """Write waypoints JSON; prefer best successful trajectories from training."""

        def make_policy(agent: CooperativeAgent):
            def _policy(position: np.ndarray) -> int:
                state = agent.discretize(position)
                return agent.greedy_action(state)

            return _policy

        data: Dict[str, List[List[float]]] = {}
        names = ["R1", "R2", "R3"]

        for idx, name in enumerate(names):
            path_opt: Optional[List[List[float]]] = None
            if best_paths and best_paths.get(name):
                path_opt = best_paths[name]

            if path_opt and len(path_opt) > 1:
                data[name] = path_opt
                continue

            # Fallback: deterministic greedy from fixed start (no oscillation noise).
            start = np.array(
                [env.cfg.start_r1, env.cfg.start_r2, env.cfg.start_r3][idx],
                dtype=float,
            )
            raw, reached, _ = env.greedy_rollout(
                make_policy(agents[idx]),
                idx,
                max_steps=min(2500, env.cfg.max_steps_per_episode),
                noise_std=0.0,
                start_pos=start,
            )
            raw = _trim_trailing_oscillation(raw)
            data[name] = raw
            if not reached:
                print(
                    f"Warning: Webots export for {name} used greedy fallback (no successful episode path recorded)."
                )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path

    def save_result(self, method: str, result: TrialResult) -> str:
        path = os.path.join(self.train_cfg.save_dir, f"{method}_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.steps_per_agent, f, indent=2)
        return path
