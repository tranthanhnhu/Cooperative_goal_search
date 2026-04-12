from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .config import EnvConfig, TrainConfig


@dataclass
class Cluster:
    mean_delta: np.ndarray
    mean_reward: float
    mean_sq_delta: np.ndarray
    mean_sq_reward: float
    count: int

    @property
    def var_delta(self) -> np.ndarray:
        return np.maximum(self.mean_sq_delta - self.mean_delta ** 2, 1e-9)

    @property
    def var_reward(self) -> float:
        return float(max(self.mean_sq_reward - self.mean_reward ** 2, 1e-9))

    def update(self, delta: np.ndarray, reward: float) -> None:
        """Incremental update following paper equations (5)-(8).

        If the cluster currently stores N-1 samples, the new sample uses
        denominator N = count + 1.
        """
        new_n = self.count + 1
        self.mean_delta = self.mean_delta + (delta - self.mean_delta) / new_n
        self.mean_sq_delta = self.mean_sq_delta + ((delta ** 2) - self.mean_sq_delta) / new_n
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / new_n
        self.mean_sq_reward = self.mean_sq_reward + ((reward ** 2) - self.mean_sq_reward) / new_n
        self.count = new_n


class ClusterModel:
    """Cluster-based stochastic model M(s,a)."""

    def __init__(self, train_cfg: TrainConfig, env_cfg: EnvConfig):
        self.train_cfg = train_cfg
        self.env_cfg = env_cfg
        self.cells: Dict[Tuple[int, int], List[Cluster]] = {}

    def _normalize_vector(self, delta: np.ndarray, reward: float) -> np.ndarray:
        dx = delta[0] / max(self.env_cfg.width, 1.0)
        dy = delta[1] / max(self.env_cfg.height, 1.0)
        rr = (reward - self.train_cfg.normalize_reward_min) / (
            self.train_cfg.normalize_reward_max - self.train_cfg.normalize_reward_min
        )
        return np.array([dx, dy, rr], dtype=float)

    def _cluster_center_vec(self, cluster: Cluster) -> np.ndarray:
        return self._normalize_vector(cluster.mean_delta, cluster.mean_reward)

    def update(self, state: int, action: int, delta: np.ndarray, reward: float) -> None:
        key = (state, action)
        vec = self._normalize_vector(delta, reward)
        clusters = self.cells.setdefault(key, [])

        if not clusters:
            clusters.append(
                Cluster(
                    mean_delta=delta.copy(),
                    mean_reward=float(reward),
                    mean_sq_delta=delta.copy() ** 2,
                    mean_sq_reward=float(reward ** 2),
                    count=1,
                )
            )
            return

        dists = [float(np.linalg.norm(vec - self._cluster_center_vec(c))) for c in clusters]
        best_idx = int(np.argmin(dists))
        if dists[best_idx] > self.train_cfg.cluster_distance_threshold:
            clusters.append(
                Cluster(
                    mean_delta=delta.copy(),
                    mean_reward=float(reward),
                    mean_sq_delta=delta.copy() ** 2,
                    mean_sq_reward=float(reward ** 2),
                    count=1,
                )
            )
        else:
            clusters[best_idx].update(delta, reward)

    def get_clusters(self, state: int, action: int) -> List[Cluster]:
        return self.cells.get((state, action), [])

    def visit_count(self, state: int, action: int) -> int:
        return sum(cluster.count for cluster in self.get_clusters(state, action))

    def sample_cluster(self, state: int, action: int, rng: np.random.Generator) -> Optional[Cluster]:
        clusters = self.get_clusters(state, action)
        if not clusters:
            return None
        counts = np.array([c.count for c in clusters], dtype=float)
        probs = counts / counts.sum()
        idx = int(rng.choice(np.arange(len(clusters)), p=probs))
        return clusters[idx]

    def append_shared_clusters(self, state: int, action: int, shared_clusters: List[Cluster]) -> None:
        """Baseline for request-based sharing without statistical fusion."""
        key = (state, action)
        current = self.cells.setdefault(key, [])
        current.extend([self.clone_cluster(c) for c in shared_clusters])

    def merge_received_clusters(self, state: int, action: int, shared_clusters: List[Cluster], t_threshold: float) -> None:
        """Paper-style knowledge fusion using T-statistics."""
        key = (state, action)
        own_clusters = self.cells.setdefault(key, [])
        if not own_clusters:
            self.cells[key] = [self.clone_cluster(c) for c in shared_clusters]
            return

        for incoming in shared_clusters:
            merged = False
            for own in own_clusters:
                t_delta = np.abs(own.mean_delta - incoming.mean_delta) / np.sqrt(
                    own.var_delta / max(own.count, 1) + incoming.var_delta / max(incoming.count, 1)
                )
                t_reward = abs(own.mean_reward - incoming.mean_reward) / math.sqrt(
                    own.var_reward / max(own.count, 1) + incoming.var_reward / max(incoming.count, 1)
                )

                if np.all(t_delta <= t_threshold) and t_reward <= t_threshold:
                    total = own.count + incoming.count
                    own.mean_delta = (own.count * own.mean_delta + incoming.count * incoming.mean_delta) / total
                    own.mean_sq_delta = (own.count * own.mean_sq_delta + incoming.count * incoming.mean_sq_delta) / total
                    own.mean_reward = (own.count * own.mean_reward + incoming.count * incoming.mean_reward) / total
                    own.mean_sq_reward = (own.count * own.mean_sq_reward + incoming.count * incoming.mean_sq_reward) / total
                    own.count = total
                    merged = True
                    break
            if not merged:
                own_clusters.append(self.clone_cluster(incoming))

    @staticmethod
    def clone_cluster(cluster: Cluster) -> Cluster:
        return Cluster(
            mean_delta=cluster.mean_delta.copy(),
            mean_reward=float(cluster.mean_reward),
            mean_sq_delta=cluster.mean_sq_delta.copy(),
            mean_sq_reward=float(cluster.mean_sq_reward),
            count=int(cluster.count),
        )


class CooperativeAgent:
    def __init__(self, agent_id: int, train_cfg: TrainConfig, env_cfg: EnvConfig, rng: np.random.Generator):
        self.agent_id = agent_id
        self.train_cfg = train_cfg
        self.env_cfg = env_cfg
        self.rng = rng
        self.state_count = train_cfg.grid_x * train_cfg.grid_y
        self.q = np.zeros((self.state_count, train_cfg.actions), dtype=float)
        self.delta_q = np.zeros_like(self.q)
        self.model = ClusterModel(train_cfg, env_cfg)

    def discretize(self, position: np.ndarray) -> int:
        x = min(max(position[0], 0.0), self.env_cfg.width - 1e-6)
        y = min(max(position[1], 0.0), self.env_cfg.height - 1e-6)
        gx = min(int((x / self.env_cfg.width) * self.train_cfg.grid_x), self.train_cfg.grid_x - 1)
        gy = min(int((y / self.env_cfg.height) * self.train_cfg.grid_y), self.train_cfg.grid_y - 1)
        return gy * self.train_cfg.grid_x + gx

    def undiscretize_center(self, state: int) -> np.ndarray:
        gx = state % self.train_cfg.grid_x
        gy = state // self.train_cfg.grid_x
        x = (gx + 0.5) * self.env_cfg.width / self.train_cfg.grid_x
        y = (gy + 0.5) * self.env_cfg.height / self.train_cfg.grid_y
        return np.array([x, y], dtype=float)

    def greedy_action(self, state: int) -> int:
        qrow = self.q[state]
        maxv = float(np.max(qrow))
        idx = np.where(qrow >= maxv - 1e-12)[0]
        return int(idx[0])

    def select_action_epsilon_greedy(self, state: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.train_cfg.actions))
        return self.greedy_action(state)

    def select_action_for_replay(self, state: int) -> int:
        # Paper Eq. (11): action selection based on update error and random exploration.
        scores = (1.0 - self.train_cfg.replay_lambda) * np.exp(self.delta_q[state]) + self.train_cfg.replay_lambda * self.rng.random(self.train_cfg.actions)
        return int(np.argmax(scores))

    def direct_q_update(self, state: int, action: int, reward: float, next_state: int) -> None:
        old_q = self.q[state, action]
        td_target = reward + self.train_cfg.gamma * np.max(self.q[next_state])
        new_q = (1.0 - self.train_cfg.alpha) * old_q + self.train_cfg.alpha * td_target
        self.q[state, action] = new_q
        self.delta_q[state, action] = new_q - old_q

    def update_from_real_experience(self, state: int, action: int, reward: float, next_state: int, pos: np.ndarray, next_pos: np.ndarray) -> None:
        self.direct_q_update(state, action, reward, next_state)
        delta = next_pos - pos
        self.model.update(state, action, delta, reward)

    def request_shared_information(self, state: int, action: int, teammates: List["CooperativeAgent"], method: str) -> None:
        received: List[Cluster] = []
        for teammate in teammates:
            if teammate.agent_id == self.agent_id:
                continue
            if teammate.model.visit_count(state, action) >= self.train_cfg.visit_threshold:
                received.extend([ClusterModel.clone_cluster(c) for c in teammate.model.get_clusters(state, action)])

        if not received:
            return

        if method == "request_sharing":
            self.model.append_shared_clusters(state, action, received)
        elif method == "proposed":
            self.model.merge_received_clusters(state, action, received, self.train_cfg.share_t_threshold)

    def planning_replay(self, teammates: List["CooperativeAgent"], method: str) -> None:
        for _ in range(self.train_cfg.planning_steps):
            sim_state = int(self.rng.integers(0, self.state_count))
            sim_pos = self.undiscretize_center(sim_state)

            for _ in range(self.train_cfg.planning_depth):
                action = self.select_action_for_replay(sim_state)
                if method in {"request_sharing", "proposed"} and self.model.visit_count(sim_state, action) < self.train_cfg.visit_threshold:
                    self.request_shared_information(sim_state, action, teammates, method)

                cluster = self.model.sample_cluster(sim_state, action, self.rng)
                if cluster is None:
                    break

                next_pos = sim_pos + cluster.mean_delta
                next_pos[0] = np.clip(next_pos[0], 0.0, self.env_cfg.width)
                next_pos[1] = np.clip(next_pos[1], 0.0, self.env_cfg.height)
                next_state = self.discretize(next_pos)
                self.direct_q_update(sim_state, action, cluster.mean_reward, next_state)
                sim_pos = next_pos
                sim_state = next_state
