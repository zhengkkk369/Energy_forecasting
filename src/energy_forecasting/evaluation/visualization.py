from __future__ import annotations

from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


@dataclass
class Plotter:
    """Produces diagnostic plots for online metrics."""

    def plot_error_curve(self, errors: List[float]) -> None:
        plt.figure(figsize=(8, 3))
        plt.plot(errors, label="MAE")
        plt.xlabel("Step")
        plt.ylabel("Error")
        plt.title("Online MAE")
        plt.legend()
        plt.tight_layout()

    def plot_spike_recovery(self, spikes: List[float], recoveries: List[int]) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].bar(range(len(spikes)), spikes)
        ax[0].set_title("Delta Spike")
        ax[1].bar(range(len(recoveries)), recoveries)
        ax[1].set_title("Recovery Time")
        plt.tight_layout()
