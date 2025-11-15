#!/usr/bin/env python3
"""
Model definitions for car image binary classification.
"""

from typing import Tuple
import torch
import torch.nn as nn
from model_arch_lib.model_shallow import CNNClassifier

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
    if model_name.lower() == 'cnn':
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

