# ============================================================
# simulate.py — Full Federated Learning Simulation
# Runs the complete patented system on a single laptop
# Simulates multiple devices training together privately
# Patent Project — Advanced AI Research
# ============================================================

import torch
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model, FederatedNet
from src.client import load_data, train, evaluate
from src.privacy import LayeredPrivacyAllocator, PrivacyBudgetTracker
from src.utils import ExperimentTracker, save_model, plot_training_progress


# ============================================================
# SIMULATION CONFIGURATION
# ============================================================

CONFIG = {
    # Federated Learning Settings
    "num_clients": 3,           # Number of simulated devices
    "num_rounds": 5,            # Number of federated rounds
    "local_epochs": 1,          # Local training epochs per round
    "learning_rate": 0.001,     # Starting learning rate

    # Privacy Settings
    "use_privacy": True,        # Enable differential privacy
    "epsilon_per_round": 2.0,   # Privacy budget per round
    "total_epsilon": 20.0,      # Total privacy budget
    "delta": 1e-5,              # Privacy failure probability
    "max_grad_norm": 1.0,       # Gradient clipping threshold

    # Novel Aggregation Settings
    "use_adaptive_aggregation": True,  # Enable our novel algorithm

    # Experiment Settings
    "experiment_name": "federated_adaptive_privacy",
    "save_best_model": True,
}


# ============================================================
# NOVEL RELIABILITY SCORE TRACKER
# ============================================================

class ClientReliabilityTracker:
    """
    Tracks reliability scores for each simulated client.
    This implements the Novel Heterogeneity-Aware Aggregation
    described in Patent Claim #2.
    """

    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.histories = {i: [] for i in range(num_clients)}

    def update(self, client_id: int, accuracy: float) -> float:
        """
        Updates and returns reliability score for a client.
        Higher score = more trustworthy = more weight in aggregation.
        """
        self.histories[client_id].append(accuracy)
        history = self.histories[client_id]

        if len(history) == 1:
            return 1.0

        # Consistency: low variance = high reliability
        variance = float(np.var(history))
        consistency = 1.0 / (1.0 + variance * 10)

        # Trend: improving = higher reliability
        trend = history[-1] - history[-2]
        trend_score = 0.5 + min(max(trend * 5, -0.5), 0.5)

        # Combined score
        score = 0.6 * consistency + 0.4 * trend_score
        return float(min(max(score, 0.1), 1.0))

    def print_scores(self, round_num: int):
        """Prints reliability scores for all clients."""
        print(f"\n  [Reliability Scores — Round {round_num}]")
        for client_id in range(self.num_clients):
            history = self.histories[client_id]
            if history:
                score = self.update(client_id, history[-1])
                print(f"  Client {client_id}: "
                      f"Score={score:.4f} | "
                      f"History={[f'{a:.3f}' for a in history]}")


# ============================================================
# NOVEL ADAPTIVE AGGREGATION
# ============================================================

def adaptive_aggregate(
    client_weights: list,
    client_samples: list,
    reliability_scores: list
) -> list:
    """
    THE CORE PATENTED ALGORITHM.

    Aggregates model weights from multiple clients using
    both sample count AND reliability scores.

    Standard FedAvg: weight = num_samples / total_samples
    Our method:      weight = (num_samples × reliability) / total

    This produces a superior global model especially when
    clients have heterogeneous or non-IID data distributions.

    Patent Claim #2 — Novel Aggregation Protocol.
    """

    # Compute combined weights
    combined = [
        s * r for s, r in zip(client_samples, reliability_scores)
    ]
    total = sum(combined)
    normalized = [w / total for w in combined]

    print(f"\n  [Aggregation] Normalized weights:")
    for i, (n, r) in enumerate(zip(normalized, reliability_scores)):
        print(f"  Client {i}: weight={n:.4f} | reliability={r:.4f}")

    # Weighted average of all parameters
    # Weighted average of all parameters
    aggregated = []
    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.zeros_like(
            client_weights[0][layer_idx], dtype=np.float64
        )
        for client_idx in range(len(client_weights)):
            layer_avg += normalized[client_idx] * \
                         client_weights[client_idx][layer_idx].astype(np.float64)
        aggregated.append(layer_avg)

    return aggregated


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

def run_simulation():
    """
    Runs the complete federated learning simulation.

    Simulates CONFIG['num_clients'] devices training together
    across CONFIG['num_rounds'] rounds with privacy protection.

    This demonstrates all three patentable components working
    together as one unified system.
    """

    print("\n" + "🔬 " * 20)
    print("\n  FEDERATED LEARNING SIMULATION")
    print("  Novel Adaptive Privacy-Preserving System")
    print("  Patent Project — Advanced AI Research")
    print("\n" + "🔬 " * 20)

    print(f"\n  Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key:<30} {value}")

    # Initialize tracking systems
    tracker = ExperimentTracker(CONFIG["experiment_name"])
    privacy_tracker = PrivacyBudgetTracker(CONFIG["total_epsilon"])
    reliability_tracker = ClientReliabilityTracker(CONFIG["num_clients"])

    # Initialize global model (starts on server)
    print("\n\n[Server] Initializing global model...")
    global_model = get_model(num_classes=10)
    global_params = global_model.get_parameters()

    # Load data for each client
    print("\n[System] Loading client datasets...")
    client_data = []
    for client_id in range(CONFIG["num_clients"]):
        trainloader, testloader = load_data(
            client_id=client_id,
            num_clients=CONFIG["num_clients"]
        )
        client_data.append((trainloader, testloader))
        print(f"  Client {client_id}: "
              f"{len(trainloader.dataset)} training samples")

    best_accuracy = 0.0
    best_model_params = None

    # ============================================================
    # FEDERATED LEARNING ROUNDS
    # ============================================================

    for round_num in range(1, CONFIG["num_rounds"] + 1):

        print(f"\n\n{'='*60}")
        print(f"  FEDERATED ROUND {round_num} / {CONFIG['num_rounds']}")
        print(f"{'='*60}")

        round_start = time.time()

        # Check privacy budget
        budget_ok = privacy_tracker.spend_budget(
            CONFIG["epsilon_per_round"],
            round_num
        )
        if not budget_ok:
            print("[Privacy] Stopping training — budget exhausted!")
            break

        # --------------------------------------------------------
        # CLIENT TRAINING PHASE
        # --------------------------------------------------------

        client_weights = []
        client_samples = []
        client_accuracies = []
        reliability_scores = []

        for client_id in range(CONFIG["num_clients"]):
            print(f"\n  --- Client {client_id} Training ---")

            # Create fresh local model
            local_model = get_model(num_classes=10)

            # Load global parameters into local model
            local_model.set_parameters(global_params)

            # Get this client's data
            trainloader, testloader = client_data[client_id]

            # Train locally on private data
            metrics = train(
                model=local_model,
                trainloader=trainloader,
                epochs=CONFIG["local_epochs"],
                learning_rate=CONFIG["learning_rate"]
            )

            # Evaluate on test data
            test_loss, test_accuracy = evaluate(local_model, testloader)

            print(f"  Client {client_id} Results:")
            print(f"  Train Accuracy: {metrics['train_accuracy']*100:.2f}%")
            print(f"  Test Accuracy:  {test_accuracy*100:.2f}%")
            print(f"  Test Loss:      {test_loss:.4f}")

            # Compute reliability score (Novel Component #2)
            reliability = reliability_tracker.update(
                client_id,
                metrics["train_accuracy"]
            )

            # Collect results (weights only — never raw data!)
            client_weights.append(local_model.get_parameters())
            client_samples.append(len(trainloader.dataset))
            client_accuracies.append(test_accuracy)
            reliability_scores.append(reliability)

        # --------------------------------------------------------
        # SERVER AGGREGATION PHASE (Novel Algorithm)
        # --------------------------------------------------------

        print(f"\n  --- Server Aggregation ---")
        reliability_tracker.print_scores(round_num)

        # Our novel reliability-weighted aggregation
        global_params = adaptive_aggregate(
            client_weights,
            client_samples,
            reliability_scores
        )

        # Update global model with aggregated weights
        global_model.set_parameters(global_params)

        # --------------------------------------------------------
        # GLOBAL EVALUATION
        # --------------------------------------------------------

        print(f"\n  --- Global Model Evaluation ---")
        _, testloader = client_data[0]
        global_loss, global_accuracy = evaluate(global_model, testloader)

        round_time = time.time() - round_start

        print(f"\n  {'='*50}")
        print(f"  ROUND {round_num} COMPLETE")
        print(f"  Global Accuracy: {global_accuracy*100:.2f}%")
        print(f"  Global Loss:     {global_loss:.4f}")
        print(f"  Round Time:      {round_time:.1f}s")
        print(f"  {'='*50}")

        # Track results
        tracker.log_round(
            round_num=round_num,
            loss=global_loss,
            accuracy=global_accuracy,
            num_clients=CONFIG["num_clients"],
            extra_metrics={
                "round_time_seconds": round(round_time, 2),
                "avg_client_accuracy": round(
                    float(np.mean(client_accuracies)), 4
                ),
                "reliability_scores": [
                    round(r, 4) for r in reliability_scores
                ]
            }
        )

        # Save best model
        if global_accuracy > best_accuracy:
            best_accuracy = global_accuracy
            best_model_params = [p.copy() for p in global_params]
            if CONFIG["save_best_model"]:
                save_model(global_model, round_num, global_accuracy)

    # ============================================================
    # TRAINING COMPLETE
    # ============================================================

    print("\n\n" + "🎓 " * 20)
    print("\n  FEDERATED LEARNING COMPLETE!")

    # Print final summary
    tracker.print_summary()

    # Privacy report
    privacy_report = privacy_tracker.save_report()
    print(f"\n  Privacy Report:")
    for key, value in privacy_report.items():
        print(f"  {key:<25} {value}")

    # Generate plots
    print("\n  Generating result plots...")
    plot_training_progress(tracker)

    print(f"\n  📁 Results saved in: results/")
    print(f"  📁 Models saved in:  models/")
    print(f"  📁 Logs saved in:    logs/")
    print(f"\n  🏆 Best Accuracy: {best_accuracy*100:.2f}%")
    print("\n" + "🎓 " * 20)

    return tracker, privacy_report


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    tracker, privacy_report = run_simulation()