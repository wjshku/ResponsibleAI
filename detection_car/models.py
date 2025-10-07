#!/usr/bin/env python3
"""
Model definitions for car image binary classification.
"""

from typing import Tuple
import torch
import torch.nn as nn
from utils import plot_training_curves



class VanillaClassifier(nn.Module):
    """Multi-layer perceptron classifier that includes basic image preprocessing and flattening.

    Expects input tensors of shape (batch, height, width, channels).
    """
    
    def __init__(self, input_size: int, hidden_size: int = 512, dropout: float = 0.5, target_size: Tuple[int, int] = (224, 224)):
        super(VanillaClassifier, self).__init__()
        self.target_size = target_size
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Accept either (N, C, H, W) tensors (from torchvision) or (N, H, W, C) arrays
        # Ensure float dtype and scale only if input seems to be uint8
        if isinstance(x, torch.Tensor):
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            else:
                x = x.float()
            # Determine layout
            if x.ndim == 4 and x.shape[1] in (1, 3):
                # Already NCHW
                x_nchw = x
            elif x.ndim == 4 and x.shape[-1] in (1, 3):
                # NHWC -> NCHW
                x_nchw = x.permute(0, 3, 1, 2)
            else:
                # Fallback: try to reshape/permute conservatively (no-op)
                x_nchw = x
        else:
            # Non-tensor input (e.g., numpy array). Convert to tensor and arrange to NCHW.
            x = torch.as_tensor(x)
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            else:
                x = x.float()
            if x.ndim == 4 and x.shape[-1] in (1, 3):
                x_nchw = x.permute(0, 3, 1, 2)
            else:
                x_nchw = x

        # Resize to target spatial size if needed
        if x_nchw.shape[2:] != self.target_size:
            x_nchw = torch.nn.functional.interpolate(x_nchw, size=self.target_size, mode='bilinear', align_corners=False)

        # Flatten to (N, C*H*W)
        batch_size = x_nchw.shape[0]
        x_flat = x_nchw.reshape(batch_size, -1)
        return self.classifier(x_flat)


class CNNClassifier(nn.Module):
    """Convolutional Neural Network for binary classification.

    hidden_size controls the width of the classifier MLP after the conv backbone.
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 1, dropout: float = 0.5, hidden_size: int = 256):
        super(CNNClassifier, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224x224 -> 112x112
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            # Fifth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
        )
        
        # Classifier head (configurable width)
        reduced_size = max(hidden_size // 4, 16)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, reduced_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(reduced_size, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Accept either (N, C, H, W) tensors (from torchvision) or (N, H, W, C) arrays
        if isinstance(x, torch.Tensor):
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            else:
                x = x.float()
            # Determine layout
            if x.ndim == 4 and x.shape[1] in (1, 3):
                x_nchw = x
            elif x.ndim == 4 and x.shape[-1] in (1, 3):
                x_nchw = x.permute(0, 3, 1, 2)
            else:
                x_nchw = x
        else:
            x = torch.as_tensor(x)
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            else:
                x = x.float()
            if x.ndim == 4 and x.shape[-1] in (1, 3):
                x_nchw = x.permute(0, 3, 1, 2)
            else:
                x_nchw = x

        # Apply convolutional features
        x_features = self.features(x_nchw)
        
        # Apply classifier
        x_out = self.classifier(x_features)
        
        return x_out


def get_model(model_name: str, input_size: int = None, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model ('linear', 'cnn', 'resnet')
        input_size: Input size for linear models (height * width * channels)
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Model instance
    """
    if model_name.lower() in ('vanilla'):
        if input_size is None:
            raise ValueError("input_size must be provided for linear/mlp model")
        return VanillaClassifier(input_size=input_size, **kwargs)
    
    elif model_name.lower() == 'cnn':
        return CNNClassifier(**kwargs)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: 'linear', 'mlp', 'cnn'")


# --------- Scikit-learn models (classical ML) ---------
def get_sklearn_model(model_name: str, **kwargs):
    """Factory for scikit-learn models: 'logreg' and 'svm'."""
    try:
        # Lazy imports to avoid hard dependency when unused
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC
    except Exception as e:
        raise ImportError("scikit-learn is required for 'logreg' or 'svm' models. Install with: uv pip install scikit-learn") from e
    name = model_name.lower()
    if name in ("logreg", "logistic", "logistic_regression"):
        # Reasonable defaults for high-dimensional sparse-like flattened pixels
        return LogisticRegression(max_iter=200, n_jobs=-1, **kwargs)
    if name in ("svm", "svc"):
        # Use linear kernel to scale better; can switch to RBF if desired
        return SVC(kernel="linear", probability=True, **kwargs)
    raise ValueError("Unknown sklearn model. Use 'logreg' or 'svm'.")


# --------- Convenience plotting helper ---------
def plot_losses(train_losses, val_losses=None, title: str = "Training and Validation Loss", save_path: str = None, show: bool = True):
    """Thin wrapper to call `utils.plot_training_curves`."""
    return plot_training_curves(train_losses, val_losses=val_losses, title=title, save_path=save_path, show=show)
