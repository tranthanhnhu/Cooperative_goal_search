from __future__ import annotations

from typing import Dict, List
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: List[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_robot_curves(result_files: Dict[str, str], output_path: str, smoothing_window: int = 15) -> str:
    method_data = {}
    for method, path in result_files.items():
        with open(path, "r", encoding="utf-8") as f:
            method_data[method] = json.load(f)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    robots = ["robot_1", "robot_2", "robot_3"]
    titles = ["Robot 1", "Robot 2", "Robot 3"]
    display_names = {
        "dyna_no_sharing": "Dyna-Q (without sharing)",
        "raw_sharing": "Dyna-Q (with sharing experiences)",
        "request_sharing": "Sharing Under Request",
        "proposed": "Proposed Sharing Method",
    }

    for ax, robot, title in zip(axes, robots, titles):
        for method, data in method_data.items():
            smoothed = moving_average(data[robot], smoothing_window)
            ax.plot(smoothed, label=display_names.get(method, method), linewidth=1.8)
        ax.set_title(title)
        ax.set_ylabel("Average #steps")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Episodes")
    fig.suptitle("Cooperative Goal Search Learning Curves", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
