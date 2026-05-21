# ============================================================
# utils.py — Utility Functions
# Experiment Tracking, Model Saving, Visualization
# Patent Project — Advanced AI Research
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# EXPERIMENT TRACKER
# ============================================================

class ExperimentTracker:
    """
    Tracks all metrics across federated learning rounds.
    Saves everything to disk for patent documentation.

    Every experiment run creates a timestamped record —
    critical for establishing invention priority date.
    """

    def __init__(self, experiment_name: str = "federated_experiment"):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.rounds: List[Dict] = []
        self.best_accuracy = 0.0
        self.best_round = 0

        # Create results directory
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        self.log_file = f"logs/{experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"

        print(f"\n[Tracker] Experiment: {experiment_name}")
        print(f"[Tracker] Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Tracker] Log file: {self.log_file}")

    def log_round(
        self,
        round_num: int,
        loss: float,
        accuracy: float,
        num_clients: int,
        extra_metrics: Optional[Dict] = None
    ):
        """Logs metrics for one federated round."""

        round_data = {
            "round": round_num,
            "loss": round(loss, 6),
            "accuracy": round(accuracy, 6),
            "accuracy_percent": round(accuracy * 100, 2),
            "num_clients": num_clients,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).seconds
        }

        if extra_metrics:
            round_data.update(extra_metrics)

        self.rounds.append(round_data)

        # Track best
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_round = round_num
            print(f"[Tracker] 🏆 New best! Round {round_num}: {accuracy*100:.2f}%")

        # Save to disk after every round
        self._save_log()

        return round_data

    def _save_log(self):
        """Saves experiment log to JSON file."""
        log_data = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "best_accuracy": self.best_accuracy,
            "best_round": self.best_round,
            "total_rounds": len(self.rounds),
            "rounds": self.rounds
        }
        with open(self.log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    def get_summary(self) -> Dict:
        """Returns experiment summary."""
        if not self.rounds:
            return {}

        losses = [r["loss"] for r in self.rounds]
        accuracies = [r["accuracy"] for r in self.rounds]

        return {
            "experiment_name": self.experiment_name,
            "total_rounds": len(self.rounds),
            "best_accuracy": f"{self.best_accuracy*100:.2f}%",
            "best_round": self.best_round,
            "final_accuracy": f"{accuracies[-1]*100:.2f}%",
            "final_loss": f"{losses[-1]:.4f}",
            "improvement": f"{(accuracies[-1] - accuracies[0])*100:.2f}%",
            "duration_seconds": (datetime.now() - self.start_time).seconds
        }

    def print_summary(self):
        """Prints a beautiful experiment summary."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("  EXPERIMENT SUMMARY")
        print("=" * 60)
        for key, value in summary.items():
            print(f"  {key:<25} {value}")
        print("=" * 60)


# ============================================================
# MODEL SAVER
# ============================================================

def save_model(
    model: nn.Module,
    round_num: int,
    accuracy: float,
    save_dir: str = "models"
) -> str:
    """
    Saves model checkpoint to disk.
    Creates timestamped filename for experiment tracking.
    """
    os.makedirs(save_dir, exist_ok=True)

    filename = (
        f"{save_dir}/federated_model_"
        f"round{round_num:03d}_"
        f"acc{accuracy*100:.1f}pct.pt"
    )

    torch.save({
        "round": round_num,
        "accuracy": accuracy,
        "model_state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat()
    }, filename)

    print(f"[Model] Saved: {filename}")
    return filename


def load_model(model: nn.Module, filepath: str) -> Tuple[nn.Module, Dict]:
    """Loads a saved model checkpoint."""
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Model] Loaded: {filepath}")
    print(f"[Model] Round: {checkpoint['round']}, "
          f"Accuracy: {checkpoint['accuracy']*100:.2f}%")
    return model, checkpoint


# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_progress(
    tracker: ExperimentTracker,
    save_path: str = "results/training_progress.png"
):
    """
    Creates a professional visualization of training progress.
    This plot will be included in the patent application.
    """
    if not tracker.rounds:
        print("[Plot] No data to plot yet.")
        return

    rounds = [r["round"] for r in tracker.rounds]
    losses = [r["loss"] for r in tracker.rounds]
    accuracies = [r["accuracy_percent"] for r in tracker.rounds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Federated Learning Training Progress\n"
        f"Novel Adaptive Aggregation Strategy",
        fontsize=14, fontweight="bold"
    )

    # Plot 1: Loss curve
    ax1.plot(rounds, losses, "b-o", linewidth=2,
             markersize=8, label="Global Loss")
    ax1.fill_between(rounds, losses, alpha=0.1, color="blue")
    ax1.set_xlabel("Federated Round", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Global Model Loss per Round", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curve
    ax2.plot(rounds, accuracies, "g-o", linewidth=2,
             markersize=8, label="Global Accuracy")
    ax2.fill_between(rounds, accuracies, alpha=0.1, color="green")
    ax2.axhline(
        y=max(accuracies), color="red",
        linestyle="--", alpha=0.5,
        label=f"Best: {max(accuracies):.1f}%"
    )
    ax2.set_xlabel("Federated Round", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Global Model Accuracy per Round", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Training progress saved to {save_path}")
    return save_path


def plot_privacy_budget(
    privacy_log: List[Dict],
    total_budget: float,
    save_path: str = "results/privacy_budget.png"
):
    """
    Visualizes privacy budget consumption over rounds.
    Shows the privacy-utility tradeoff — key for patent docs.
    """
    if not privacy_log:
        print("[Plot] No privacy data to plot.")
        return

    rounds = [r["round"] for r in privacy_log]
    spent = [r["total_spent"] for r in privacy_log]
    remaining = [r["remaining"] for r in privacy_log]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.stackplot(
        rounds,
        spent, remaining,
        labels=["Budget Spent (ε)", "Budget Remaining"],
        colors=["#FF6B6B", "#4ECDC4"],
        alpha=0.8
    )

    ax.axhline(
        y=total_budget, color="black",
        linestyle="--", linewidth=2,
        label=f"Total Budget (ε={total_budget})"
    )

    ax.set_xlabel("Training Round", fontsize=12)
    ax.set_ylabel("Privacy Budget (ε)", fontsize=12)
    ax.set_title(
        "Privacy Budget Consumption\n"
        "Novel Layered Differential Privacy Allocator",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Privacy budget saved to {save_path}")


# ============================================================
# SYSTEM INFO PRINTER
# ============================================================

def print_system_info():
    """Prints system information for documentation."""
    print("\n" + "=" * 60)
    print("  SYSTEM INFORMATION")
    print("=" * 60)
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  Device:     {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)

    print_system_info()

    # Test experiment tracker
    tracker = ExperimentTracker("test_experiment")

    # Simulate some rounds
    fake_rounds = [
        (1, 0.85, 0.72),
        (2, 0.71, 0.79),
        (3, 0.62, 0.84),
        (4, 0.55, 0.87),
        (5, 0.49, 0.90),
    ]

    for round_num, loss, acc in fake_rounds:
        tracker.log_round(round_num, loss, acc, num_clients=3)
        time.sleep(0.1)

    # Print summary
    tracker.print_summary()

    # Generate plots
    plot_training_progress(tracker)

    print("\n✅ All utility functions working perfectly!")
    print("=" * 60)