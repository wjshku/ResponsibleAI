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
            'label_loss': [],  # Source label loss
            'label_target_loss': [],  # Target label loss (monitoring only)
            'domain_loss': []
        }

        self.train_accuracies = {
            'label_accuracy': {
                'source_accuracy': [],
                'target_accuracy': []
            },
            'domain_accuracy': []
        }

        self.val_losses = {
            'label_loss': {
                'source_loss': [],
                'target_loss': []
            },
            'domain_loss': []
        }

        self.val_accuracies = {
            'label_accuracy': {
                'source_accuracy': [],
                'target_accuracy': []
            },
            'domain_accuracy': []
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
                           train_label_loss: float,
                           train_label_target_loss: float,
                           train_domain_loss: float,
                           train_label_source_acc: float,
                           train_label_target_acc: float,
                           train_domain_acc: float,
                           val_label_source_loss: float,
                           val_label_target_loss: float,
                           val_label_source_acc: float,
                           val_label_target_acc: float,
                           val_domain_loss: float,
                           val_domain_accuracy: float,
                           lambda_value: float,
                           epoch_time: float):
        """
        Update metrics for a training epoch.

        Args:
            train_label_loss: Training label classification loss (source domain)
            train_label_target_loss: Training label classification loss (target domain, monitoring only)
            train_domain_loss: Training domain classification loss
            train_label_source_acc: Training label accuracy on source domain
            train_label_target_acc: Training label accuracy on target domain
            train_domain_acc: Training domain classification accuracy
            val_label_source_loss: Validation label loss on source domain
            val_label_target_loss: Validation label loss on target domain
            val_label_source_acc: Validation label accuracy on source domain
            val_label_target_acc: Validation label accuracy on target domain
            val_domain_loss: Validation domain classification loss
            val_domain_accuracy: Validation domain classification accuracy
            lambda_value: Lambda adaptation strength
            epoch_time: Time taken for the epoch
        """
        # Update training losses
        self.train_losses['label_loss'].append(train_label_loss)
        self.train_losses['label_target_loss'].append(train_label_target_loss)
        self.train_losses['domain_loss'].append(train_domain_loss)

        # Update training accuracies
        self.train_accuracies['label_accuracy']['source_accuracy'].append(train_label_source_acc)
        self.train_accuracies['label_accuracy']['target_accuracy'].append(train_label_target_acc)
        self.train_accuracies['domain_accuracy'].append(train_domain_acc)

        # Update validation losses
        self.val_losses['label_loss']['source_loss'].append(val_label_source_loss)
        self.val_losses['label_loss']['target_loss'].append(val_label_target_loss)
        self.val_losses['domain_loss'].append(val_domain_loss)

        # Update validation accuracies
        self.val_accuracies['label_accuracy']['source_accuracy'].append(val_label_source_acc)
        self.val_accuracies['label_accuracy']['target_accuracy'].append(val_label_target_acc)
        self.val_accuracies['domain_accuracy'].append(val_domain_accuracy)

        # Update lambda and timing
        self.lambda_values.append(lambda_value)
        self.epoch_times.append(epoch_time)

        # Update best accuracies
        if val_label_source_acc > self.best_source_acc:
            self.best_source_acc = val_label_source_acc

        if val_label_target_acc > self.best_target_acc:
            self.best_target_acc = val_label_target_acc
            self.best_epoch = len(self.val_accuracies['label_accuracy']['target_accuracy'])

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
            'val_losses': self.val_losses,
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
        Plot loss curves: Label losses and Domain losses.

        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.train_losses['label_loss']:
            print("Warning: No loss data available for plotting")
            return

        epochs = range(1, len(self.train_losses['label_loss']) + 1)

        plt.figure(figsize=(15, 5))

        # Plot 1: Label Losses (Source & Target, Train & Test) with Lambda
        plt.subplot(1, 2, 1)
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # Source: solid lines, Target: dashed lines
        # Train: blue, Test: red
        line1 = ax1.plot(epochs, self.train_losses['label_loss'], 'b-', label='Source Train', linewidth=2, marker='o', markersize=4)
        line2 = ax1.plot(epochs, self.val_losses['label_loss']['source_loss'], 'r-', label='Source Test', linewidth=2, marker='s', markersize=4)
        line3 = ax1.plot(epochs, self.train_losses['label_target_loss'], 'b--', label='Target Train', linewidth=2, marker='^', markersize=4)
        line4 = ax1.plot(epochs, self.val_losses['label_loss']['target_loss'], 'r--', label='Target Test', linewidth=2, marker='v', markersize=4)

        # Lambda on secondary axis
        line5 = ax2.plot(epochs, self.lambda_values, 'purple', linewidth=1.5, alpha=0.7, label='Lambda')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Lambda', color='purple')
        plt.title('Label Classification Losses & Adaptation Strength')
        ax1.grid(True, alpha=0.3)

        # Combine legends
        lines = line1 + line2 + line3 + line4 + line5
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

        # Plot 2: Domain Losses (Train & Test) with Lambda
        plt.subplot(1, 2, 2)
        ax3 = plt.gca()
        ax4 = ax3.twinx()

        # Train: blue solid, Test: red solid
        line6 = ax3.plot(epochs, self.train_losses['domain_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
        line7 = ax3.plot(epochs, self.val_losses['domain_loss'], 'r-', label='Test', linewidth=2, marker='s', markersize=4)

        # Lambda on secondary axis
        line8 = ax4.plot(epochs, self.lambda_values, 'purple', linewidth=1.5, alpha=0.7, label='Lambda')

        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax4.set_ylabel('Lambda', color='purple')
        plt.title('Domain Classification Losses & Adaptation Strength')
        ax3.grid(True, alpha=0.3)

        # Combine legends
        lines2 = line6 + line7 + line8
        labels2 = [l.get_label() for l in lines2]
        ax3.legend(lines2, labels2, loc='upper right')

        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Loss curves plot saved to {save_path}")

    def plot_accuracy_curves(self, save_path: Optional[str] = None):
        """
        Plot accuracy curves: Label classification and Domain classification.

        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.train_accuracies['label_accuracy']['source_accuracy']:
            print("Warning: No accuracy data available for plotting")
            return

        epochs = range(1, len(self.train_accuracies['label_accuracy']['source_accuracy']) + 1)

        plt.figure(figsize=(15, 5))

        # Plot 1: Label Classification Accuracy (Source & Target, Train & Test)
        # Source: solid lines, Target: dashed lines
        # Train: blue, Test: red
        plt.subplot(1, 2, 1)
        plt.plot(epochs, [acc * 100 for acc in self.train_accuracies['label_accuracy']['source_accuracy']], 'b-', label='Source Train', linewidth=2, marker='o', markersize=3)
        plt.plot(epochs, [acc * 100 for acc in self.val_accuracies['label_accuracy']['source_accuracy']], 'r-', label='Source Test', linewidth=2, marker='s', markersize=3)
        plt.plot(epochs, [acc * 100 for acc in self.train_accuracies['label_accuracy']['target_accuracy']], 'b--', label='Target Train', linewidth=2, marker='^', markersize=3)
        plt.plot(epochs, [acc * 100 for acc in self.val_accuracies['label_accuracy']['target_accuracy']], 'r--', label='Target Test', linewidth=2, marker='v', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Label Classification Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)

        # Plot 2: Domain Classification Accuracy (Train & Test)
        # Train: blue, Test: red
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [acc * 100 for acc in self.train_accuracies['domain_accuracy']], 'b-', label='Train Domain Acc', linewidth=2, marker='o', markersize=3)
        # Use the actual computed domain accuracy on test sets
        plt.plot(epochs, [acc * 100 for acc in self.val_accuracies['domain_accuracy']], 'r-', label='Test Domain Acc', linewidth=2, marker='s', markersize=3)
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Random Baseline')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Domain Classification Accuracy')
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

    def plot_test_adversarial_dynamics(self, save_path=None):
        """
        Plot test-based adversarial training dynamics with test losses and lambda schedule.
        """
        import matplotlib.pyplot as plt

        if not self.history:
            print("No training history available for plotting")
            return

        # Extract data from history
        epochs = [h['epoch'] for h in self.history]
        source_test_losses = [h.get('source_val_loss', 0) for h in self.history]
        target_test_losses = [h.get('target_val_loss', 0) for h in self.history]
        lambdas = [h.get('lambda_value', 0) for h in self.history]

        # Create plot with dual y-axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = ax.twinx()

        # Plot test losses
        line1 = ax.plot(epochs, source_test_losses, 'b-', linewidth=2, label='Source Test Loss (MNIST)', marker='o', markersize=4)
        line2 = ax.plot(epochs, target_test_losses, 'r-', linewidth=2, label='Target Test Loss (MNIST-M)', marker='s', markersize=4)

        # Plot lambda schedule
        line3 = ax2.plot(epochs, lambdas, 'g--', linewidth=2, label='Lambda (GRL Strength)', marker='^', markersize=4)

        ax.set_title('DANN Test Loss Adversarial Dynamics', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Loss', color='black')
        ax2.set_ylabel('Lambda (Gradient Reversal Strength)', color='green')

        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')

        ax.grid(True, alpha=0.3)

        # Save the plot
        if save_path is None:
            save_path = self.save_dir / "test_adversarial_dynamics.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Test adversarial dynamics plot saved to {save_path}")

    def generate_all_plots(self):
        """Generate and save all plots."""
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
        final_train_source_label_acc = self.train_accuracies['label_accuracy']['source_accuracy'][-1]
        final_train_target_label_acc = self.train_accuracies['label_accuracy']['target_accuracy'][-1]
        final_domain_acc = self.train_accuracies['domain_accuracy'][-1]
        final_val_source_label_acc = self.val_accuracies['label_accuracy']['source_accuracy'][-1]
        final_val_target_label_acc = self.val_accuracies['label_accuracy']['target_accuracy'][-1]

        print("\nFinal epoch statistics:")
        print(f"  Training Label Loss: {final_label_loss:.4f}")
        print(f"  Training Domain Loss: {final_domain_loss:.4f}")
        print(f"  Training Source Label Acc: {final_train_source_label_acc * 100:.2f}%")
        print(f"  Training Target Label Acc: {final_train_target_label_acc * 100:.2f}%")
        print(f"  Training Domain Acc: {final_domain_acc * 100:.2f}%")
        print(f"  Validation Source Label Acc: {final_val_source_label_acc * 100:.2f}%")
        print(f"  Validation Target Label Acc: {final_val_target_label_acc * 100:.2f}%")

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
            'train_label_loss': tracker.train_losses['label_loss'][-1] if tracker.train_losses['label_loss'] else None,
            'train_domain_loss': tracker.train_losses['domain_loss'][-1] if tracker.train_losses['domain_loss'] else None,
            'train_label_source_accuracy': tracker.train_accuracies['label_accuracy']['source_accuracy'][-1] if tracker.train_accuracies['label_accuracy']['source_accuracy'] else None,
            'train_label_target_accuracy': tracker.train_accuracies['label_accuracy']['target_accuracy'][-1] if tracker.train_accuracies['label_accuracy']['target_accuracy'] else None,
            'train_domain_accuracy': tracker.train_accuracies['domain_accuracy'][-1] if tracker.train_accuracies['domain_accuracy'] else None,
            'val_label_source_accuracy': tracker.val_accuracies['label_accuracy']['source_accuracy'][-1] if tracker.val_accuracies['label_accuracy']['source_accuracy'] else None,
            'val_label_target_accuracy': tracker.val_accuracies['label_accuracy']['target_accuracy'][-1] if tracker.val_accuracies['label_accuracy']['target_accuracy'] else None,
            'val_label_source_loss': tracker.val_losses['label_loss']['source_loss'][-1] if tracker.val_losses['label_loss']['source_loss'] else None,
            'val_label_target_loss': tracker.val_losses['label_loss']['target_loss'][-1] if tracker.val_losses['label_loss']['target_loss'] else None,
            'val_domain_loss': tracker.val_losses['domain_loss'][-1] if tracker.val_losses['domain_loss'] else None
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
            train_label_loss=0.5 - epoch * 0.1,
            train_domain_loss=0.8 - epoch * 0.05,
            train_label_source_acc=(85 + epoch * 2) / 100.0,  # Convert to fraction
            train_label_target_acc=0.0,                        # No target labels in training
            train_domain_acc=(50 + epoch) / 100.0,             # Convert to fraction
            val_label_source_loss=0.6 - epoch * 0.08,
            val_label_target_loss=0.9 - epoch * 0.06,
            val_label_source_acc=(82 + epoch) / 100.0,         # Convert to fraction
            val_label_target_acc=(78 + epoch) / 100.0,          # Convert to fraction
            val_domain_loss=0.0,                                # Not computed on validation
            val_domain_accuracy=(60 + epoch) / 100.0,           # Domain accuracy on validation
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
