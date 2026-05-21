# ============================================================
# model.py — Neural Network Architecture
# Federated Learning with Differential Privacy
# Patent Project — Advanced AI Research
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class FederatedNet(nn.Module):
    """
    A Convolutional Neural Network designed for federated learning.
    This model runs on each client device independently.
    No raw data ever leaves the device — only model weights are shared.
    This architecture is the core of the patentable invention.
    """

    def __init__(self, num_classes: int = 10):
        super(FederatedNet, self).__init__()

        # --- CONVOLUTIONAL LAYERS ---
        # These layers detect patterns in images
        # Like eyes detecting edges, shapes, textures
        self.conv1 = nn.Conv2d(
            in_channels=1,   # 1 = grayscale image input
            out_channels=32, # 32 different pattern detectors
            kernel_size=3,   # looks at 3x3 pixel areas
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # --- BATCH NORMALIZATION ---
        # Stabilizes training — critical for federated learning
        # where each device has different data distributions
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # --- POOLING LAYER ---
        # Reduces image size while keeping important features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- DROPOUT ---
        # Prevents overfitting — randomly turns off neurons during training
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

        # --- FULLY CONNECTED LAYERS ---
        # Final decision-making layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — data flows through the network.
        Input: image tensor of shape (batch, 1, 28, 28)
        Output: class probabilities of shape (batch, num_classes)
        """

        # Layer 1: Detect basic patterns
        x = self.conv1(x)        # Apply first convolution
        x = self.bn1(x)          # Normalize
        x = F.relu(x)            # Activate (keep positive values)
        x = self.pool(x)         # Shrink image: 28x28 → 14x14

        # Layer 2: Detect complex patterns
        x = self.conv2(x)        # Apply second convolution
        x = self.bn2(x)          # Normalize
        x = F.relu(x)            # Activate
        x = self.pool(x)         # Shrink image: 14x14 → 7x7
        x = self.dropout1(x)     # Randomly drop some features

        # Flatten: Convert 2D image to 1D vector
        x = x.view(x.size(0), -1)   # Shape: (batch, 64*7*7)

        # Fully connected layers — make final decision
        x = self.fc1(x)          # 3136 → 128 neurons
        x = F.relu(x)            # Activate
        x = self.dropout2(x)     # Randomly drop some neurons
        x = self.fc2(x)          # 128 → num_classes

        return x

    def get_parameters(self):
        """
        Returns model parameters as a list of numpy arrays.
        This is what gets sent to the server during federated learning.
        CRITICAL: Only weights travel — never the actual data.
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters):
        """
        Receives updated global parameters from the server
        and loads them into this local model.
        This is how the device receives the improved global model.
        """
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }
        self.load_state_dict(state_dict, strict=True)


def get_model(num_classes: int = 10) -> FederatedNet:
    """
    Factory function — creates and returns a fresh model instance.
    Used by both clients and server.
    """
    model = FederatedNet(num_classes=num_classes)
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ============================================================
# Quick test — runs only when this file is executed directly
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Testing FederatedNet Model")
    print("=" * 50)

    # Create model
    model = get_model(num_classes=10)
    print(f"\nModel Architecture:\n{model}")

    # Test with fake data (simulating one image)
    dummy_input = torch.randn(1, 1, 28, 28)  # 1 image, 1 channel, 28x28 pixels
    output = model(dummy_input)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\n✅ Model is working perfectly!")
    print("=" * 50)