# ============================================================
# client.py — Federated Learning Client
# Represents ONE device in the federated network
# Trains locally, shares only weights — never raw data
# Patent Project — Advanced AI Research
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple

# Import our custom model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FederatedNet, get_model


# ============================================================
# DATA LOADING
# ============================================================

def load_data(client_id: int, num_clients: int = 5):
    """
    Loads MNIST dataset and splits it across clients.
    Each client only sees its own portion of data —
    simulating real-world devices with private local data.

    Args:
        client_id: Which client this is (0, 1, 2, 3, 4)
        num_clients: Total number of clients in the system
    """

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),               # Convert image to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize pixel values
    ])

    # Download MNIST dataset (handwritten digits 0-9)
    # This downloads automatically on first run
    print(f"[Client {client_id}] Loading MNIST dataset...")
    trainset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    testset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Split training data equally among all clients
    # Each client gets its own private slice of data
    total_size = len(trainset)
    client_size = total_size // num_clients
    sizes = [client_size] * num_clients

    # Handle remainder
    sizes[-1] += total_size - sum(sizes)

    # Split the dataset
    client_datasets = random_split(
        trainset,
        sizes,
        generator=torch.Generator().manual_seed(42)
    )

    # Get this client's data only
    client_trainset = client_datasets[client_id]

    print(f"[Client {client_id}] Training samples: {len(client_trainset)}")
    print(f"[Client {client_id}] Test samples: {len(testset)}")

    # Create data loaders
    trainloader = DataLoader(
        client_trainset,
        batch_size=32,       # Process 32 images at a time
        shuffle=True,        # Shuffle data each epoch
        num_workers=0        # No parallel loading (Windows compatible)
    )
    testloader = DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    return trainloader, testloader


# ============================================================
# LOCAL TRAINING FUNCTION
# ============================================================

def train(
    model: FederatedNet,
    trainloader: DataLoader,
    epochs: int = 1,
    learning_rate: float = 0.001
) -> Dict[str, float]:
    """
    Trains the model on LOCAL data only.
    This is the core privacy-preserving step —
    data never leaves this function.

    Args:
        model: The neural network to train
        trainloader: Local private data
        epochs: How many times to go through the data
        learning_rate: How fast the model learns

    Returns:
        Dictionary with training metrics
    """

    # Define loss function — measures how wrong the model is
    criterion = nn.CrossEntropyLoss()

    # Define optimizer — adjusts weights to reduce loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4    # L2 regularization — prevents overfitting
    )

    model.train()  # Set model to training mode
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_idx, (images, labels) in enumerate(trainloader):
            # Zero out previous gradients
            optimizer.zero_grad()

            # Forward pass — make predictions
            outputs = model(images)

            # Calculate loss — how wrong were we?
            loss = criterion(outputs, labels)

            # Backward pass — calculate gradients
            loss.backward()

            # Clip gradients — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_samples += labels.size(0)

        avg_loss = epoch_loss / len(trainloader)
        accuracy = epoch_correct / epoch_samples
        total_loss += avg_loss
        correct += epoch_correct
        total_samples += epoch_samples

        print(f"  Epoch {epoch+1}/{epochs} — "
              f"Loss: {avg_loss:.4f} — "
              f"Accuracy: {accuracy:.4f}")

    return {
        "train_loss": total_loss / epochs,
        "train_accuracy": correct / total_samples
    }


# ============================================================
# LOCAL EVALUATION FUNCTION
# ============================================================

def evaluate(
    model: FederatedNet,
    testloader: DataLoader
) -> Tuple[float, float]:
    """
    Evaluates model performance on test data.
    Measures how well the model learned without seeing test data.

    Returns:
        Tuple of (loss, accuracy)
    """

    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set model to evaluation mode

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():  # No gradient calculation needed
        for images, labels in testloader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(testloader)
    accuracy = correct / total_samples

    return avg_loss, accuracy


# ============================================================
# FLOWER CLIENT CLASS — The Heart of Federated Learning
# ============================================================

class FederatedClient(fl.client.NumPyClient):
    """
    The Flower federated learning client.
    This class connects our local training to the global
    federated learning system.

    Each device runs one instance of this class.
    The server never sees the raw data — only the weights.
    This is the NOVEL PATENTABLE component.
    """

    def __init__(self, client_id: int, num_clients: int = 5):
        self.client_id = client_id
        self.model = get_model(num_classes=10)
        self.trainloader, self.testloader = load_data(
            client_id, num_clients
        )
        print(f"\n[Client {client_id}] Ready for federated learning!")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Called by server to get current model weights.
        Returns weights as numpy arrays — never raw data.
        """
        print(f"[Client {self.client_id}] Sending parameters to server...")
        return self.model.get_parameters()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Called by server to train on local data.
        1. Receive global model from server
        2. Train on private local data
        3. Send updated weights back — never the data
        """
        print(f"\n[Client {self.client_id}] === Starting Local Training ===")

        # Step 1: Load global model weights from server
        self.model.set_parameters(parameters)

        # Step 2: Get training config from server
        epochs = config.get("local_epochs", 1)
        lr = config.get("learning_rate", 0.001)

        # Step 3: Train on LOCAL private data
        metrics = train(self.model, self.trainloader, epochs, lr)

        print(f"[Client {self.client_id}] Training complete!")
        print(f"[Client {self.client_id}] Loss: {metrics['train_loss']:.4f}")
        print(f"[Client {self.client_id}] Accuracy: {metrics['train_accuracy']:.4f}")

        # Step 4: Return updated weights (NOT the data)
        return (
            self.model.get_parameters(),
            len(self.trainloader.dataset),
            metrics
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Called by server to evaluate model performance.
        Tests how well the global model works on local data.
        """
        print(f"[Client {self.client_id}] Evaluating model...")

        # Load latest global model
        self.model.set_parameters(parameters)

        # Evaluate on local test data
        loss, accuracy = evaluate(self.model, self.testloader)

        print(f"[Client {self.client_id}] Test Loss: {loss:.4f}")
        print(f"[Client {self.client_id}] Test Accuracy: {accuracy:.4f}")

        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


# ============================================================
# Quick standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Federated Client (Standalone)")
    print("=" * 50)

    # Create client 0 out of 5 total clients
    client = FederatedClient(client_id=0, num_clients=5)

    # Simulate one round of training
    print("\n--- Simulating one training round ---")
    params = client.model.get_parameters()
    updated_params, num_samples, metrics = client.fit(
        params,
        config={"local_epochs": 1, "learning_rate": 0.001}
    )

    print(f"\n✅ Client test complete!")
    print(f"Trained on {num_samples} samples")
    print(f"Final metrics: {metrics}")
    print("=" * 50)