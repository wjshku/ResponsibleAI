#!/usr/bin/env python3
"""
General utilities: model saving/loading helpers and visualization utilities.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import torch


# ---------------------- Visualization Utilities ----------------------
def visualize_transforms(image_tensor, title: str):
    """Visualize an image after transforms.

    Accepts torch tensors (any device) or numpy arrays in CHW/HWC, with optional
    ImageNet normalization. Displays RGB or grayscale as appropriate.
    """
    # Lazy import to avoid hard dependency
    import matplotlib.pyplot as plt

    image = image_tensor

    if isinstance(image_tensor, torch.Tensor):
        tensor = image_tensor.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            if tensor.shape[0] == 3:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = tensor * std + mean
            tensor = tensor.permute(1, 2, 0)
        np_img = tensor.numpy()
        if np_img.ndim == 2:
            np_img = np_img[:, :, None]
        if np.issubdtype(np_img.dtype, np.floating):
            np_img = (np_img * 255.0).clip(0, 255)
        image = np_img.astype(np.uint8)
    else:
        np_img = image_tensor
        if isinstance(np_img, np.ndarray):
            if np_img.ndim == 3 and np_img.shape[0] in (1, 3) and np_img.shape[2] not in (1, 3, 4):
                np_img = np.transpose(np_img, (1, 2, 0))
            if np_img.ndim == 2:
                np_img = np_img[:, :, None]
            if np.issubdtype(np_img.dtype, np.floating):
                np_img = (np_img * 255.0).clip(0, 255)
            image = np_img.astype(np.uint8)

    cmap = 'gray' if (isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 1) else None
    plt.figure(figsize=(6, 6))
    plt.imshow(image.squeeze() if cmap == 'gray' else image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


# ---------------------- Training Curves Plotting ----------------------
def plot_training_curves(train_losses, val_losses=None, title: str = "Training and Validation Loss", save_path: Union[str, Path] = None, show: bool = True):
    """Plot training loss (and optional validation loss) over epochs.

    Args:
        train_losses: List of training loss values per epoch.
        val_losses: Optional list of validation/test loss values per epoch.
        title: Plot title.
        save_path: Optional file path to save the figure.
        show: If True, displays the plot window.
    """
    # Lazy import to avoid hard dependency when unused
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    if val_losses is not None and len(val_losses) == len(train_losses):
        plt.plot(epochs, val_losses, label="Validation Loss", marker="s")
    elif val_losses is not None:
        # Length mismatch; still plot but warn in console
        print("Warning: val_losses length doesn't match train_losses; plotting available points.")
        v_epochs = list(range(1, len(val_losses) + 1))
        plt.plot(v_epochs, val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved training curves to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

# ---------------------- Model Utilities (from model_utils.py) ----------------------

def create_model_directory(base_dir: str = "models") -> Path:
    """Create a timestamped model directory for organized storage."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = Path(base_dir) / f"model_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_sklearn_model(model, model_name: str, results: Dict[str, Any], 
                      config: Dict[str, Any], model_dir: Path = None) -> Path:
    """
    Save scikit-learn model with metadata and results.
    """
    from joblib import dump
    if model_dir is None:
        model_dir = create_model_directory()
    model_path = model_dir / f"{model_name}_model.joblib"
    dump(model, model_path)
    metadata = {
        'model_name': model_name,
        'model_type': 'sklearn',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'config': config,
        'results': results,
        'model_file': str(model_path.name)
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    results_path = model_dir / "results.txt"
    with open(results_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Type: Scikit-learn\n")
        f.write(f"Timestamp: {metadata['timestamp']}\n")
        f.write(f"Test Accuracy: {results.get('test_accuracy', 'N/A'):.2f}%\n")
        f.write(f"Training Time: {results.get('training_time', 'N/A'):.2f}s\n")
        f.write(f"Train Samples: {results.get('train_samples', 'N/A')}\n")
        f.write(f"Test Samples: {results.get('test_samples', 'N/A')}\n")
        f.write(f"Features: {results.get('features', 'N/A')}\n")
        f.write(f"\nConfiguration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
    print(f"Model saved to: {model_dir}")
    print(f"  - Model file: {model_path.name}")
    print(f"  - Metadata: metadata.json")
    print(f"  - Results: results.txt")
    return model_dir


def save_pytorch_model(model, model_name: str, results: Dict[str, Any], 
                      config: Dict[str, Any], model_dir: Path = None) -> Path:
    """Save PyTorch model with metadata and results."""
    if model_dir is None:
        model_dir = create_model_directory()
    model_path = model_dir / f"{model_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    checkpoint_path = model_dir / f"{model_name}_checkpoint.pth"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': results,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, checkpoint_path)
    metadata = {
        'model_name': model_name,
        'model_type': 'pytorch',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'config': config,
        'results': results,
        'model_file': str(model_path.name),
        'checkpoint_file': str(checkpoint_path.name),
        'model_parameters': sum(p.numel() for p in model.parameters())
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    results_path = model_dir / "results.txt"
    with open(results_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Type: PyTorch\n")
        f.write(f"Timestamp: {metadata['timestamp']}\n")
        f.write(f"Parameters: {metadata['model_parameters']:,}\n")
        f.write(f"Best Test Accuracy: {results.get('best_test_acc', 'N/A'):.2f}%\n")
        f.write(f"Final Train Loss: {results.get('train_losses', [0])[-1]:.4f}\n")
        f.write(f"Final Train Accuracy: {results.get('train_accuracies', [0])[-1]:.2f}%\n")
        f.write(f"Final Test Loss: {results.get('test_losses', [0])[-1]:.4f}\n")
        f.write(f"Final Test Accuracy: {results.get('test_accuracies', [0])[-1]:.2f}%\n")
        f.write(f"Epochs: {config.get('num_epochs', 'N/A')}\n")
        f.write(f"Learning Rate: {config.get('learning_rate', 'N/A')}\n")
        f.write(f"Batch Size: {config.get('batch_size', 'N/A')}\n")
        f.write(f"Train Samples: {results.get('train_samples', 'N/A')}\n")
        f.write(f"Test Samples: {results.get('test_samples', 'N/A')}\n")
        f.write(f"\nConfiguration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
    print(f"Model saved to: {model_dir}")
    print(f"  - Model file: {model_path.name}")
    print(f"  - Checkpoint: {checkpoint_path.name}")
    print(f"  - Metadata: metadata.json")
    print(f"  - Results: results.txt")
    return model_dir


def load_pytorch_model(model_dir: Path, model_class, device='cpu'):
    """Load a PyTorch model from a saved directory."""
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    model_path = model_dir / metadata['model_file']
    model = model_class(**metadata['config'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model, metadata


def list_saved_models(base_dir: str = "models") -> list:
    """List all saved models in the models directory."""
    models_dir = Path(base_dir)
    if not models_dir.exists():
        return []
    model_dirs = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "metadata.json").exists():
            try:
                with open(model_dir / "metadata.json", 'r') as f:
                    metadata = json.load(f)
                model_dirs.append({
                    'path': model_dir,
                    'name': metadata.get('model_name', 'unknown'),
                    'type': metadata.get('model_type', 'unknown'),
                    'timestamp': metadata.get('timestamp', 'unknown'),
                    'accuracy': metadata.get('results', {}).get('test_accuracy', 'N/A')
                })
            except Exception as e:
                print(f"Warning: Could not read metadata for {model_dir}: {e}")
    return sorted(model_dirs, key=lambda x: x['timestamp'], reverse=True)


def print_model_summary(model_dir: Path):
    """Print a summary of a saved model."""
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"No metadata found in {model_dir}")
        return
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"\nModel Summary: {model_dir.name}")
    print("=" * 50)
    print(f"Name: {metadata.get('model_name', 'N/A')}")
    print(f"Type: {metadata.get('model_type', 'N/A')}")
    print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
    results = metadata.get('results', {})
    if metadata.get('model_type') == 'sklearn':
        print(f"Test Accuracy: {results.get('test_accuracy', 'N/A'):.2f}%")
        print(f"Training Time: {results.get('training_time', 'N/A'):.2f}s")
    else:
        print(f"Best Test Accuracy: {results.get('best_test_acc', 'N/A'):.2f}%")
        print(f"Parameters: {metadata.get('model_parameters', 'N/A'):,}")
    print(f"Files:")
    for file in model_dir.iterdir():
        if file.is_file():
            print(f"  - {file.name}")
