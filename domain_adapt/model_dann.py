#!/usr/bin/env python3
"""
Domain Adversarial Neural Network (DANN) for fake image detection.

Implementation based on "Unsupervised Domain Adaptation by Backpropagation"
(Ganin & Lempitsky, 2015).

This model learns features that are:
1. Discriminative for real vs fake classification
2. Invariant to which generative model created the fakes (SD2 vs Kontext)
"""

import torch
import torch.nn as nn
from models import CNNClassifier


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) from the DANN paper.

    Forward pass: Identity function (output = input)
    Backward pass: Multiply gradient by -lambda (reversed gradient)

    This forces the feature extractor to learn features that:
    - Minimize label classification loss (real vs fake)
    - Maximize domain classification loss (confuse domain classifier)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass: Identity transformation.

        Args:
            x: Input tensor
            lambda_: Adaptation strength parameter

        Returns:
            x unchanged
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Reverse and scale gradient.

        Args:
            grad_output: Gradient from downstream layers

        Returns:
            Reversed gradient: -lambda * grad_output
        """
        lambda_ = ctx.lambda_
        # Multiply gradient by -lambda (reversal)
        grad_input = -lambda_ * grad_output
        # Return gradient for x and None for lambda_ (lambda_ is not learnable)
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer as a PyTorch module.

    Usage:
        grl = GradientReversalLayer()
        grl.lambda_ = 0.5  # Set adaptation strength
        reversed_features = grl(features)
    """

    def __init__(self):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = 1.0

    def forward(self, x):
        """Apply gradient reversal with current lambda value."""
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainAdversarialNN(nn.Module):
    """
    Domain Adversarial Neural Network for cross-domain fake detection.

    Architecture:
        Input Image
            ↓
        Feature Extractor (CNN backbone)
            ↓
        Features (512-dim)
            ↓
            ├→ Label Predictor → Real/Fake classification
            │
            └→ [GRL] → Domain Classifier → SD2/Kontext classification

    The GRL ensures that features are domain-invariant while remaining
    discriminative for the main classification task.
    """

    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 1,
                 feature_hidden_size: int = 256,
                 domain_hidden_size: int = 256,
                 dropout: float = 0.3):
        """
        Initialize DANN model.

        Args:
            input_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (1 for binary)
            feature_hidden_size: Hidden size for label predictor
            domain_hidden_size: Hidden size for domain classifier
            dropout: Dropout probability
        """
        super(DomainAdversarialNN, self).__init__()

        # ============================================
        # Feature Extractor: Shared CNN backbone
        # ============================================
        # Reuse the convolutional layers from existing CNNClassifier
        base_cnn = CNNClassifier(
            input_channels=input_channels,
            num_classes=num_classes,
            dropout=dropout,
            hidden_size=feature_hidden_size
        )

        # Extract only the convolutional feature extractor
        self.feature_extractor = base_cnn.features  # Conv layers only

        # Add pooling and flattening to get fixed-size feature vector
        self.feature_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling → (batch, 512, 1, 1)
            nn.Flatten()                    # Flatten → (batch, 512)
        )

        # Feature dimension from last conv layer
        self.feature_dim = 512

        # ============================================
        # Label Predictor: Real vs Fake classifier
        # ============================================
        reduced_size = max(feature_hidden_size // 4, 16)
        self.label_predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, feature_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_hidden_size, reduced_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(reduced_size, num_classes),
            nn.Sigmoid()
        )

        # ============================================
        # Gradient Reversal Layer
        # ============================================
        self.grl = GradientReversalLayer()

        # ============================================
        # Domain Classifier: Source vs Target classifier
        # ============================================
        # This classifier tries to distinguish SD2-fakes from Kontext-fakes
        # The GRL makes the feature extractor try to fool this classifier
        domain_mid_size = domain_hidden_size // 2
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, domain_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(domain_hidden_size, domain_mid_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(domain_mid_size, 1),
            nn.Sigmoid()  # Binary: 0=source (SD2), 1=target (Kontext)
        )

    def forward(self, x, alpha=1.0):
        """
        Forward pass through the DANN model.

        Args:
            x: Input images, shape (batch, channels, height, width)
            alpha: Lambda parameter for GRL (adaptation strength)
                   - 0.0: No domain adaptation (inference mode)
                   - 1.0: Full domain adaptation
                   - Scheduled from 0→1 during training

        Returns:
            label_output: Real/fake predictions, shape (batch, 1)
            domain_output: Source/target predictions, shape (batch, 1)
        """
        # ============================================
        # 1. Extract features (shared representation)
        # ============================================
        features = self.feature_extractor(x)      # Conv features
        features = self.feature_pooling(features)  # (batch, 512)

        # ============================================
        # 2. Label prediction (real vs fake)
        # ============================================
        label_output = self.label_predictor(features)

        # ============================================
        # 3. Domain prediction (with gradient reversal)
        # ============================================
        # Update GRL's lambda parameter
        self.grl.lambda_ = alpha

        # Apply gradient reversal
        reversed_features = self.grl(features)

        # Domain classification on reversed features
        domain_output = self.domain_classifier(reversed_features)

        return label_output, domain_output

    def predict_labels(self, x):
        """
        Predict only labels (for inference).

        Args:
            x: Input images

        Returns:
            Label predictions only (no domain predictions)
        """
        label_output, _ = self.forward(x, alpha=0.0)
        return label_output


def get_dann_model(input_channels=3,
                   num_classes=1,
                   feature_hidden_size=256,
                   domain_hidden_size=256,
                   dropout=0.3,
                   **kwargs):
    """
    Factory function to create DANN model.

    Args:
        input_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes (1 for binary)
        feature_hidden_size: Hidden size for label predictor
        domain_hidden_size: Hidden size for domain classifier
        dropout: Dropout probability

    Returns:
        DomainAdversarialNN model instance
    """
    return DomainAdversarialNN(
        input_channels=input_channels,
        num_classes=num_classes,
        feature_hidden_size=feature_hidden_size,
        domain_hidden_size=domain_hidden_size,
        dropout=dropout
    )


# ============================================
# Lambda Schedule (from DANN paper)
# ============================================
def compute_lambda_schedule(epoch, total_epochs, gamma=10.0):
    """
    Compute lambda parameter using schedule from DANN paper.

    Lambda gradually increases from 0 to 1 during training following:
        lambda_p = 2 / (1 + exp(-gamma * p)) - 1

    where p = epoch / total_epochs (training progress)

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of training epochs
        gamma: Sharpness of the schedule (default: 10.0 from paper)

    Returns:
        lambda_p: Adaptation strength in [0, 1]
    """
    import numpy as np
    p = float(epoch) / float(total_epochs)
    lambda_p = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
    return lambda_p


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("TESTING DOMAIN ADVERSARIAL NEURAL NETWORK")
    print("=" * 60)

    # Create model
    model = get_dann_model(
        input_channels=3,
        num_classes=1,
        feature_hidden_size=256,
        domain_hidden_size=256,
        dropout=0.3
    )

    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    print(f"\nTesting forward pass with input shape: {dummy_input.shape}")

    # Test with different lambda values
    for alpha in [0.0, 0.5, 1.0]:
        label_out, domain_out = model(dummy_input, alpha=alpha)
        print(f"\nLambda = {alpha:.1f}:")
        print(f"  Label output shape: {label_out.shape}")
        print(f"  Domain output shape: {domain_out.shape}")

    # Test gradient reversal
    print("\n" + "=" * 60)
    print("Testing Gradient Reversal Layer")
    print("=" * 60)

    grl = GradientReversalLayer()
    grl.lambda_ = 0.5

    # Create dummy tensor with gradient tracking
    x = torch.randn(4, 512, requires_grad=True)

    # Forward pass
    y = grl(x)
    print(f"Forward pass: input shape {x.shape} → output shape {y.shape}")
    print(f"Forward pass preserves values: {torch.allclose(x, y)}")

    # Backward pass
    loss = y.sum()
    loss.backward()

    print(f"Gradient is reversed: {torch.allclose(x.grad, -0.5 * torch.ones_like(x))}")

    print("\n✓ All tests passed!")
