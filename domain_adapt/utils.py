#!/usr/bin/env python3
"""
Utility functions for DANN training progress tracking and visualization.

This module provides functions to:
1. Track training metrics during DANN training
2. Generate plots for loss curves, accuracy curves, and gradient analysis
3. Save results to timestamped subfolders in models_mnist (e.g., model_dann_20251110_123241)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class TrainingTracker:
    """
    Class to track training metrics during DANN training.

    Tracks losses, accuracies, gradients, and other metrics over training epochs.
    """

    def __init__(self, save_dir: str = "./models_mnist"):
        """
        Initialize the training tracker.

        Args:
            save_dir: Base directory for saving results (will create timestamped subfolder)
        """
        # Create timestamped subfolder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(save_dir)
        self.save_dir = self.base_dir / f"model_dann_{timestamp}"

        # Create directories
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # Training metrics
        self.train_losses = {
            'label_loss': [],
            'domain_loss': [],
            'total_loss': []
        }

        self.train_accuracies = {
            'label_accuracy': [],
            'domain_accuracy': [],
            'source_domain_accuracy': [],
            'target_domain_accuracy': []
        }

        self.val_accuracies = {
            'source_accuracy': [],
            'target_accuracy': []
        }

        # Gradient tracking
        self.gradients = {
            'feature_extractor': [],
            'label_classifier': [],
            'domain_classifier': []
        }

        # Lambda values
        self.lambda_values = []

        # Epoch timing
        self.epoch_times = []

        # Best accuracies
        self.best_source_acc = 0.0
        self.best_target_acc = 0.0
        self.best_epoch = 0

    def update_epoch_metrics(self,
                           label_loss: float,
                           domain_loss: float,
                           total_loss: float,
                           label_accuracy: float,
                           domain_accuracy: float,
                           source_domain_accuracy: float,
                           target_domain_accuracy: float,
                           source_val_accuracy: float,
                           target_val_accuracy: float,
                           lambda_value: float,
                           epoch_time: float):
        """
        Update metrics for a training epoch.

        Args:
            label_loss: Label classification loss
            domain_loss: Domain classification loss
            total_loss: Total loss
            label_accuracy: Label classification accuracy
            domain_accuracy: Domain classification accuracy
            source_domain_accuracy: Source domain accuracy
            target_domain_accuracy: Target domain accuracy
            source_val_accuracy: Source validation accuracy
            target_val_accuracy: Target validation accuracy
            lambda_value: Lambda adaptation strength
            epoch_time: Time taken for the epoch
        """
        # Update losses
        self.train_losses['label_loss'].append(label_loss)
        self.train_losses['domain_loss'].append(domain_loss)
        self.train_losses['total_loss'].append(total_loss)

        # Update accuracies
        self.train_accuracies['label_accuracy'].append(label_accuracy)
        self.train_accuracies['domain_accuracy'].append(domain_accuracy)
        self.train_accuracies['source_domain_accuracy'].append(source_domain_accuracy)
        self.train_accuracies['target_domain_accuracy'].append(target_domain_accuracy)

        # Update validation accuracies
        self.val_accuracies['source_accuracy'].append(source_val_accuracy)
        self.val_accuracies['target_accuracy'].append(target_val_accuracy)

        # Update lambda and timing
        self.lambda_values.append(lambda_value)
        self.epoch_times.append(epoch_time)

        # Update best accuracies
        if source_val_accuracy > self.best_source_acc:
            self.best_source_acc = source_val_accuracy

        if target_val_accuracy > self.best_target_acc:
            self.best_target_acc = target_val_accuracy
            self.best_epoch = len(self.val_accuracies['target_accuracy'])

    def track_gradients(self, model: nn.Module):
        """
        Track gradients for different components of the model.

        Args:
            model: The DANN model after backpropagation
        """
        def get_grad_norm(layer_name: str, layer: nn.Module) -> float:
            """Get the L2 norm of gradients for a layer."""
            total_norm = 0.0
            param_count = 0
            for param in layer.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            return total_norm ** 0.5 if param_count > 0 else 0.0

        # Track gradients for different components
        feature_grad = get_grad_norm('feature_extractor', model.feature)
        label_grad = get_grad_norm('label_classifier', model.class_classifier)
        domain_grad = get_grad_norm('domain_classifier', model.domain_classifier)

        self.gradients['feature_extractor'].append(feature_grad)
        self.gradients['label_classifier'].append(label_grad)
        self.gradients['domain_classifier'].append(domain_grad)

    def save_metrics(self, filename: str = "training_metrics.json"):
        """
        Save all training metrics to a JSON file.

        Args:
            filename: Name of the metrics file
        """
        metrics = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'gradients': self.gradients,
            'lambda_values': self.lambda_values,
            'epoch_times': self.epoch_times,
            'best_accuracies': {
                'source': self.best_source_acc,
                'target': self.best_target_acc,
                'epoch': self.best_epoch
            }
        }

        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Training metrics saved to {filepath}")

    def plot_loss_curves(self, save_path: Optional[str] = None):
        """
        Plot 1: Label classifier loss vs Domain classifier loss.

        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.train_losses['label_loss']:
            print("Warning: No loss data available for plotting")
            return

        epochs = range(1, len(self.train_losses['label_loss']) + 1)

        plt.figure(figsize=(12, 5))

        # Plot 1: Losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses['label_loss'], 'b-', label='Label Loss', linewidth=2)
        plt.plot(epochs, self.train_losses['domain_loss'], 'r-', label='Domain Loss', linewidth=2)
        plt.plot(epochs, self.train_losses['total_loss'], 'g-', label='Total Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Lambda schedule
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.lambda_values, 'purple', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Lambda')
        plt.title('Domain Adaptation Strength (λ)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Loss curves plot saved to {save_path}")

    def plot_accuracy_curves(self, save_path: Optional[str] = None):
        """
        Plot 2: Label classifier accuracy vs Domain classifier accuracy.

        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.train_accuracies['label_accuracy']:
            print("Warning: No accuracy data available for plotting")
            return

        epochs = range(1, len(self.train_accuracies['label_accuracy']) + 1)

        plt.figure(figsize=(15, 5))

        # Plot label accuracy
        plt.subplot(1, 3, 1)
        plt.plot(epochs, [acc * 100 for acc in self.train_accuracies['label_accuracy']], 'b-', label='Train Label Acc', linewidth=2)
        plt.plot(epochs, [acc * 100 for acc in self.val_accuracies['source_accuracy']], 'b--', label='Val Source Acc', linewidth=2)
        plt.plot(epochs, [acc * 100 for acc in self.val_accuracies['target_accuracy']], 'b:', label='Val Target Acc', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Label Classification Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)

        # Plot domain accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs, [acc * 100 for acc in self.train_accuracies['domain_accuracy']], 'r-', label='Train Domain Acc', linewidth=2)
        plt.plot(epochs, [acc * 100 for acc in self.train_accuracies['source_domain_accuracy']], 'r--', label='Train Source Domain', linewidth=2)
        plt.plot(epochs, [acc * 100 for acc in self.train_accuracies['target_domain_accuracy']], 'r:', label='Train Target Domain', linewidth=2)
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Random (50%)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Domain Classification Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)

        # Plot combined view
        plt.subplot(1, 3, 3)
        plt.plot(epochs, [acc * 100 for acc in self.val_accuracies['source_accuracy']], 'blue', label='Source Domain', linewidth=2)
        plt.plot(epochs, [acc * 100 for acc in self.val_accuracies['target_accuracy']], 'red', label='Target Domain', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Source vs Target Domain Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)

        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / "accuracy_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Accuracy curves plot saved to {save_path}")

    def plot_gradient_analysis(self, save_path: Optional[str] = None):
        """
        Plot 3: Feature extractor, label classifier, and domain classifier gradients.

        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.gradients['feature_extractor']:
            print("Warning: No gradient data available for plotting")
            return

        epochs = range(1, len(self.gradients['feature_extractor']) + 1)

        plt.figure(figsize=(15, 10))

        # Plot gradient norms over time
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.gradients['feature_extractor'], 'green', label='Feature Extractor', linewidth=2)
        plt.plot(epochs, self.gradients['label_classifier'], 'blue', label='Label Classifier', linewidth=2)
        plt.plot(epochs, self.gradients['domain_classifier'], 'red', label='Domain Classifier', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Gradient L2 Norm')
        plt.title('Gradient Norms Over Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization

        # Plot gradient ratios
        plt.subplot(2, 2, 2)
        feature_grads = np.array(self.gradients['feature_extractor'])
        label_grads = np.array(self.gradients['label_classifier'])
        domain_grads = np.array(self.gradients['domain_classifier'])

        # Avoid division by zero
        eps = 1e-8
        label_to_feature = label_grads / (feature_grads + eps)
        domain_to_feature = domain_grads / (feature_grads + eps)
        domain_to_label = domain_grads / (label_grads + eps)

        plt.plot(epochs, label_to_feature, 'purple', label='Label/Feature Ratio', linewidth=2)
        plt.plot(epochs, domain_to_feature, 'orange', label='Domain/Feature Ratio', linewidth=2)
        plt.plot(epochs, domain_to_label, 'brown', label='Domain/Label Ratio', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Ratio')
        plt.title('Gradient Ratios (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Plot gradient statistics
        plt.subplot(2, 2, 3)
        grad_data = [self.gradients['feature_extractor'], self.gradients['label_classifier'], self.gradients['domain_classifier']]
        labels = ['Feature Extractor', 'Label Classifier', 'Domain Classifier']

        plt.boxplot(grad_data, labels=labels)
        plt.ylabel('Gradient L2 Norm')
        plt.title('Gradient Distribution (Box Plot)')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Plot gradient correlation
        plt.subplot(2, 2, 4)
        # Compute correlation matrix
        grad_matrix = np.column_stack([
            self.gradients['feature_extractor'],
            self.gradients['label_classifier'],
            self.gradients['domain_classifier']
        ])
        corr_matrix = np.corrcoef(grad_matrix.T)

        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks([0, 1, 2], labels, rotation=45)
        plt.yticks([0, 1, 2], labels)
        plt.title('Gradient Correlation Matrix')

        # Add correlation values as text
        for i in range(3):
            for j in range(3):
                plt.text(j, i, '.2f', ha='center', va='center', fontweight='bold')

        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / "gradient_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Gradient analysis plot saved to {save_path}")

    def generate_all_plots(self):
        """Generate and save all three plots."""
        print("\n📊 Generating training plots...")
        self.plot_loss_curves()
        self.plot_accuracy_curves()
        self.plot_gradient_analysis()
        print("✅ All plots generated successfully!")

    def print_summary(self):
        """Print a summary of the training results."""
        if not self.train_losses['label_loss']:
            print("No training data available")
            return

        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)

        n_epochs = len(self.train_losses['label_loss'])

        print(f"Total epochs trained: {n_epochs}")
        print(f"Average epoch time: {sum(self.epoch_times)/len(self.epoch_times):.1f}s")
        print(f"Best source accuracy: {self.best_source_acc * 100:.2f}%")
        print(f"Best target accuracy: {self.best_target_acc * 100:.2f}%")
        print(f"Best epoch: {self.best_epoch}")

        # Final epoch statistics
        final_label_loss = self.train_losses['label_loss'][-1]
        final_domain_loss = self.train_losses['domain_loss'][-1]
        final_label_acc = self.train_accuracies['label_accuracy'][-1]
        final_domain_acc = self.train_accuracies['domain_accuracy'][-1]
        final_source_acc = self.val_accuracies['source_accuracy'][-1]
        final_target_acc = self.val_accuracies['target_accuracy'][-1]

        print("\nFinal epoch statistics:")
        print(f"  Label Loss: {final_label_loss:.4f}")
        print(f"  Domain Loss: {final_domain_loss:.4f}")
        print(f"  Label Accuracy: {final_label_acc * 100:.2f}%")
        print(f"  Domain Accuracy: {final_domain_acc * 100:.2f}%")
        print(f"  Source Val Accuracy: {final_source_acc * 100:.2f}%")
        print(f"  Target Val Accuracy: {final_target_acc * 100:.2f}%")

        # Best accuracies
        print("\nBest accuracies:")
        print(f"  Best Source Accuracy: {self.best_source_acc * 100:.2f}%")
        print(f"  Best Target Accuracy: {self.best_target_acc * 100:.2f}%")

        print("="*60)


# ============================================
# Convenience Functions for Training
# ============================================

def create_tracker(save_dir: str = "./models_mnist") -> TrainingTracker:
    """
    Create a new training tracker.

    Args:
        save_dir: Directory to save results

    Returns:
        TrainingTracker instance
    """
    return TrainingTracker(save_dir)


def save_training_summary(tracker: TrainingTracker, model: nn.Module,
                         config: Dict[str, Any], filename: str = "training_summary.json"):
    """
    Save a comprehensive training summary.

    Args:
        tracker: TrainingTracker instance
        model: Trained model
        config: Training configuration
        filename: Summary filename
    """
    summary = {
        'config': config,
        'best_accuracies': {
            'source': tracker.best_source_acc,
            'target': tracker.best_target_acc,
            'epoch': tracker.best_epoch
        },
        'final_metrics': {
            'label_loss': tracker.train_losses['label_loss'][-1] if tracker.train_losses['label_loss'] else None,
            'domain_loss': tracker.train_losses['domain_loss'][-1] if tracker.train_losses['domain_loss'] else None,
            'label_accuracy': tracker.train_accuracies['label_accuracy'][-1] if tracker.train_accuracies['label_accuracy'] else None,
            'domain_accuracy': tracker.train_accuracies['domain_accuracy'][-1] if tracker.train_accuracies['domain_accuracy'] else None,
            'source_val_accuracy': tracker.val_accuracies['source_accuracy'][-1] if tracker.val_accuracies['source_accuracy'] else None,
            'target_val_accuracy': tracker.val_accuracies['target_accuracy'][-1] if tracker.val_accuracies['target_accuracy'] else None
        },
        'model_info': {
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }

    filepath = tracker.save_dir / filename
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    tracker = create_tracker()

    # Simulate some training data
    for epoch in range(5):
        tracker.update_epoch_metrics(
            label_loss=0.5 - epoch * 0.1,
            domain_loss=0.8 - epoch * 0.05,
            total_loss=1.3 - epoch * 0.15,
            label_accuracy=85 + epoch * 2,
            domain_accuracy=50 + epoch,
            source_domain_accuracy=45 + epoch,
            target_domain_accuracy=55 + epoch,
            source_val_accuracy=82 + epoch,
            target_val_accuracy=78 + epoch,
            lambda_value=epoch * 0.2,
            epoch_time=10.5
        )

    # Generate plots
    tracker.generate_all_plots()

    # Save metrics
    tracker.save_metrics()

    # Print summary
    tracker.print_summary()

    print(f"Example plots and metrics saved to {tracker.save_dir}/")
