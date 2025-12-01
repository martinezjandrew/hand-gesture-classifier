import torch.nn as nn


class HandMLP(nn.Module):
    """
    Multi-Layer Perceptron for hand gesture classification.

    Architecture:
    - Input: 42 features (21 landmarks × 2 coordinates)
    - Hidden Layer 1: 256 neurons (BatchNorm + GELU + Dropout)
    - Hidden Layer 2: 128 neurons (BatchNorm + GELU + Dropout)
    - Output: 34 classes
    """

    def __init__(self, input_size=42, num_classes=34):
        """
        Initialize the MLP model.

        Args:
            input_size (int): Number of input features
            num_classes (int): Number of output classes
        """
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            # Output layer
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.net(x)
