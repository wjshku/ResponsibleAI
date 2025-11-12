#!/usr/bin/env python3
"""
Abstract base class for DANN (Domain Adversarial Neural Network) models.

This module provides an abstract base class that defines the standard DANN architecture:
- Feature Extractor (shared backbone)
- Label Classifier (task-specific)
- Domain Classifier (domain discriminator with gradient reversal)

Subclass this to create new DANN models for different datasets or architectures.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


# ============================================
# Gradient Reversal Layer (GRL)
# ============================================
class ReverseLayerF(torch.autograd.Function):
    """
    Gradient Reversal Layer from the DANN paper.

    Forward pass: Identity function
    Backward pass: Multiply gradient by -alpha (reversal)

    This layer enables adversarial training by reversing gradients during
    backpropagation, forcing the feature extractor to learn domain-invariant features.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        """
        Forward pass: Identity function.

        Args:
            ctx: Context object for storing values
            x: Input features
            alpha: Gradient reversal strength (0=no reversal, 1=full reversal)

        Returns:
            x: Same as input (identity function)
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Multiply gradient by -alpha.

        Args:
            ctx: Context object containing alpha
            grad_output: Gradient from next layer

        Returns:
            Reversed gradient, None (for alpha)
        """
        output = grad_output.neg() * ctx.alpha
        return output, None


# ============================================
# Abstract Base Class for DANN Models
# ============================================
class AbstractDANN(nn.Module, ABC):
    """
    Abstract base class for Domain Adversarial Neural Network (DANN) models.

    This class defines the standard DANN architecture with three components:
    1. Feature Extractor: Shared backbone that extracts features from input
    2. Label Classifier: Task-specific classifier (e.g., digit classification)
    3. Domain Classifier: Domain discriminator (source vs target)

    The forward pass follows this flow:
        Input → Feature Extractor → Features
        Features → Label Classifier → Label Predictions
        Features → [GRL] → Domain Classifier → Domain Predictions

    Subclass this and implement:
        - _build_feature_extractor(): Build the feature extraction backbone
        - _build_label_classifier(): Build the label classification head
        - _build_domain_classifier(): Build the domain classification head
        - _get_feature_dim(): Return the flattened feature dimension

    Example:
        class MyDANNModel(AbstractDANN):
            def _build_feature_extractor(self):
                return nn.Sequential(...)
            
            def _build_label_classifier(self):
                return nn.Sequential(...)
            
            def _build_domain_classifier(self):
                return nn.Sequential(...)
            
            def _get_feature_dim(self):
                return 800  # Flattened feature dimension
    """

    def __init__(self):
        """
        Initialize the DANN model.

        Subclasses should call super().__init__() and then build components.
        """
        super(AbstractDANN, self).__init__()

        # Build the three main components
        self.feature_extractor = self._build_feature_extractor()
        self.class_classifier = self._build_label_classifier()
        self.domain_classifier = self._build_domain_classifier()

    @abstractmethod
    def _build_feature_extractor(self) -> nn.Module:
        """
        Build the feature extraction backbone.

        This should be a shared network that extracts features from input images
        and outputs flattened features (2D tensor: batch, feature_dim).
        The output should be suitable for both label and domain classification.

        The feature extractor should handle:
        1. Feature extraction (e.g., CNN layers)
        2. Flattening to 2D tensor (batch, feature_dim)

        Returns:
            nn.Module: Feature extraction network that outputs (batch, feature_dim)
        """
        pass

    @abstractmethod
    def _build_label_classifier(self) -> nn.Module:
        """
        Build the label classification head.

        This network takes flattened features and outputs label predictions.
        Typically ends with LogSoftmax for NLLLoss compatibility.

        Returns:
            nn.Module: Label classification network
        """
        pass

    @abstractmethod
    def _build_domain_classifier(self) -> nn.Module:
        """
        Build the domain classification head.

        This network takes features (after gradient reversal) and outputs
        domain predictions (source vs target). Typically ends with LogSoftmax.

        Returns:
            nn.Module: Domain classification network
        """
        pass

    @abstractmethod
    def _get_feature_dim(self) -> int:
        """
        Get the flattened feature dimension.

        This is needed to properly flatten features before passing to classifiers.
        Should return the size of features after flattening (e.g., channels * height * width).

        Returns:
            int: Flattened feature dimension
        """
        pass

    def forward(self, input_data: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DANN model.

        Standard DANN forward procedure:
        1. Extract and flatten features from input (via feature extractor)
        2. Apply gradient reversal to features (for domain classification)
        3. Predict labels from original features
        4. Predict domain from reversed features

        Args:
            input_data: Input images (batch, channels, height, width)
            alpha: Gradient reversal strength (0=no reversal, 1=full reversal)
                   Typically scheduled during training (0 → 1)

        Returns:
            Tuple of (class_output, domain_output):
            - class_output: Label predictions (batch, num_classes) with LogSoftmax
            - domain_output: Domain predictions (batch, 2) with LogSoftmax
        """
        # Extract and flatten features (feature extractor handles both)
        feature = self.feature_extractor(input_data)

        # Apply gradient reversal for domain classification
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        # Class prediction (digit/task classification)
        class_output = self.class_classifier(feature)

        # Domain prediction (source vs target)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extractor module.

        Returns:
            Feature extractor network
        """
        return self.feature_extractor

    def get_label_classifier(self) -> nn.Module:
        """
        Get the label classifier module.

        Returns:
            Label classifier network
        """
        return self.class_classifier

    def get_domain_classifier(self) -> nn.Module:
        """
        Get the domain classifier module.

        Returns:
            Domain classifier network
        """
        return self.domain_classifier


# ============================================
# Example Implementation Template
# ============================================
class ExampleDANNModel(AbstractDANN):
    """
    Example implementation template for creating new DANN models.

    Copy this class and modify to create your own DANN model.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        """
        Initialize example DANN model.

        Args:
            input_channels: Number of input channels (e.g., 3 for RGB)
            num_classes: Number of label classes (e.g., 10 for digits)
        """
        self.input_channels = input_channels
        self.num_classes = num_classes
        super().__init__()

    def _build_feature_extractor(self) -> nn.Module:
        """
        Build feature extractor (template - implement your architecture).
        
        Should output flattened features (batch, feature_dim).
        """
        # Example: Simple CNN with flattening
        # return nn.Sequential(
        #     nn.Conv2d(self.input_channels, 64, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     # ... more layers ...
        #     nn.Flatten(),  # Flatten to (batch, feature_dim)
        # )
        raise NotImplementedError("Subclass ExampleDANNModel and implement _build_feature_extractor()")

    def _build_label_classifier(self) -> nn.Module:
        """
        Build label classifier (template - implement your architecture).
        """
        feature_dim = self._get_feature_dim()
        # Example: MLP
        # return nn.Sequential(
        #     nn.Linear(feature_dim, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, self.num_classes),
        #     nn.LogSoftmax(dim=1)
        # )
        raise NotImplementedError("Subclass ExampleDANNModel and implement _build_label_classifier()")

    def _build_domain_classifier(self) -> nn.Module:
        """
        Build domain classifier (template - implement your architecture).
        """
        feature_dim = self._get_feature_dim()
        # Example: MLP for binary classification (source vs target)
        # return nn.Sequential(
        #     nn.Linear(feature_dim, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 2),
        #     nn.LogSoftmax(dim=1)
        # )
        raise NotImplementedError("Subclass ExampleDANNModel and implement _build_domain_classifier()")

    def _get_feature_dim(self) -> int:
        """
        Get flattened feature dimension (template - implement based on your architecture).
        """
        # Example: channels * height * width after feature extractor
        # return 64 * 4 * 4  # Assuming 64 channels, 4x4 spatial size
        raise NotImplementedError("Subclass ExampleDANNModel and implement _get_feature_dim()")



if __name__ == "__main__":
    print("Abstract DANN Model Base Classes")
    print("=" * 50)
    print("\nTo create a new DANN model:")
    print("1. Subclass AbstractDANN")
    print("2. Implement required abstract methods:")
    print("   - _build_feature_extractor() - should output flattened features (batch, feature_dim)")
    print("   - _build_label_classifier()")
    print("   - _build_domain_classifier()")
    print("   - _get_feature_dim()")
    print("\nThe forward() method is provided by the base class.")
    print("\nExample:")
    print("""
    class MyDANNModel(AbstractDANN):
        def _build_feature_extractor(self):
            # Feature extractor should output flattened features
            return nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),  # Flatten to (batch, feature_dim)
                # Or: nn.AdaptiveAvgPool2d(1), nn.Flatten()
            )
        
        def _build_label_classifier(self):
            feature_dim = self._get_feature_dim()
            return nn.Sequential(...)
        
        def _build_domain_classifier(self):
            feature_dim = self._get_feature_dim()
            return nn.Sequential(...)
        
        def _get_feature_dim(self):
            return 800
    """)

