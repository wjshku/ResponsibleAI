#!/usr/bin/env python3
"""
Training script for neural network models (MLP, CNN).
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import copy
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import (
    CarDDDataset,
    create_dataloader,
    get_train_transforms,
    get_eval_transforms,
)
from models import get_model
from utils import save_pytorch_model, plot_losses, plot_training_curves

def train_epoch(model, train_loader, criterion, optimizer, device, track_time=False):
    """Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cuda/cpu)
        track_time: If True, track and print timing information for each batch
    """
    model.train()
    running_loss = 0.0
    running_grad_norm = 0.0
    grad_norm_count = 0  # Track how many times we computed grad norm
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Create batch progress bar
    batch_pbar = tqdm(train_loader, desc="Training", leave=False, unit="batch")
    
    # Calculate gradient norm every N batches (reduce overhead)
    GRAD_NORM_FREQ = 10  # Compute grad norm every 10 batches

    # Track end time of last step for image loading timing (only if tracking)
    last_step_end_time = time.time() if track_time else None

    for batch_idx, (images, labels) in enumerate(batch_pbar):
        # Track image loading time (from end of last step to current step)
        if track_time:
            current_step_start_time = time.time()
            image_load_time = current_step_start_time - last_step_end_time if last_step_end_time else 0
            if batch_idx > 0:  # Skip first batch as there's no previous step
                print(f"Batch {batch_idx}: Image loading time: {image_load_time*1000:.2f}ms")

        # Track GPU transfer time
        if track_time:
            gpu_transfer_start = time.time()
        images, labels = images.to(device), labels.to(device)
        if track_time:
            gpu_transfer_time = time.time() - gpu_transfer_start
            print(f"Batch {batch_idx}: GPU transfer time: {gpu_transfer_time*1000:.2f}ms")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        outputs = outputs.squeeze()
        
        # Ensure outputs and labels have the same shape
        if outputs.dim() == 0:  # scalar
            outputs = outputs.unsqueeze(0)
        if labels.dim() == 0:  # scalar
            labels = labels.unsqueeze(0)
        
        # Calculate loss
        loss = criterion(outputs, labels.float())
        
        # Track gradient calculation time
        if track_time:
            grad_start_time = time.time()
        # Backward pass
        loss.backward()
        if track_time:
            grad_time = time.time() - grad_start_time
            print(f"Batch {batch_idx}: Gradient calculation time: {grad_time*1000:.2f}ms")

        # Track gradient norm for monitoring (only every N batches to reduce overhead)
        if (batch_idx + 1) % GRAD_NORM_FREQ == 0:
            # More efficient gradient norm calculation
            grad_norm_sq = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
            grad_norm = grad_norm_sq ** 0.5
            running_grad_norm += grad_norm
            grad_norm_count += 1

        optimizer.step()

        # Statistics
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Calculate confusion matrix components
        tp = ((predicted == 1) & (labels == 1)).sum().item()
        fp = ((predicted == 1) & (labels == 0)).sum().item()
        fn = ((predicted == 0) & (labels == 1)).sum().item()
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn

        # Update progress bar with current batch metrics
        current_loss = running_loss / (len(batch_pbar) if len(batch_pbar) > 0 else 1)
        current_acc = 100.0 * correct / total if total > 0 else 0
        batch_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{current_acc:.1f}%"
        })
        
        # Record end time of current step for next iteration's image loading timing
        if track_time:
            last_step_end_time = time.time()

    batch_pbar.close()

    epoch_loss = running_loss / len(train_loader)
    # Average grad norm over the batches where it was computed
    epoch_grad_norm = running_grad_norm / grad_norm_count if grad_norm_count > 0 else 0.0
    epoch_acc = 100.0 * correct / total

    # Calculate precision, recall, and F1-score
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / max(1e-8, precision + recall)

    return {
        'loss': epoch_loss,
        'grad_norm': epoch_grad_norm,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Create batch progress bar for evaluation
    batch_pbar = tqdm(test_loader, desc="Evaluating", leave=False, unit="batch")

    with torch.no_grad():
        for images, labels in batch_pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            outputs = outputs.squeeze()
            
            # Ensure outputs and labels have the same shape
            if outputs.dim() == 0:  # scalar
                outputs = outputs.unsqueeze(0)
            if labels.dim() == 0:  # scalar
                labels = labels.unsqueeze(0)
            
            # Calculate loss
            loss = criterion(outputs, labels.float())
            
            # Statistics
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate confusion matrix components
            tp = ((predicted == 1) & (labels == 1)).sum().item()
            fp = ((predicted == 1) & (labels == 0)).sum().item()
            fn = ((predicted == 0) & (labels == 1)).sum().item()
            
            true_positives += tp
            false_positives += fp
            false_negatives += fn

            # Update progress bar with current batch metrics
            current_loss = running_loss / (len(batch_pbar) if len(batch_pbar) > 0 else 1)
            current_acc = 100.0 * correct / total if total > 0 else 0
            batch_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_acc:.1f}%"
            })

    batch_pbar.close()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100.0 * correct / total

    # Calculate precision, recall, and F1-score
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / max(1e-8, precision + recall)

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def main():
    """Main training function for neural network models."""
    parser = argparse.ArgumentParser(description='Train neural network for car damage detection')
    parser.add_argument('--domain', type=str, default='sd2', choices=['sd2', 'kontext', 'qwen'],
                       help='Training domain (default: sd2)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size for training (default: None = full dataset)')
    parser.add_argument('--target_size', type=int, default=224,
                       help='Image target size (default: 224, use 224 for speed, 512 for quality)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs (default: 20)')
    parser.add_argument('--model', type=str, default='cnn', choices=['vanilla', 'cnn'],
                       help='Model type (default: cnn)')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden size for model (default: 256)')
    parser.add_argument('--track_time', action='store_true', default=False,
                       help='Track and print timing information for each batch (default: False)')

    args = parser.parse_args()

    print("=" * 60)
    print("CAR IMAGE BINARY CLASSIFICATION - NEURAL NETWORKS")
    print("=" * 60)
    print(f"Domain: {args.domain.upper()}")
    print(f"Sample size: {args.sample_size or 'Full dataset'}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Model: {args.model}")
    print(f"Track time: {args.track_time}")
    print("=" * 60)

    train_domain = args.domain
    sample_size = args.sample_size
    
    # Configuration
    config = {
        'train_domain': train_domain,
        'sample_size': sample_size,
        'target_size': (args.target_size, args.target_size),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.epochs,
        'model_name': args.model,
        'hidden_size': args.hidden_size,
        'dropout': 0.2,
        'random_state': 42
    }

    print(f"Random seed: {config['random_state']}")
    print(f"Dropout: {config['dropout']}")
    print("=" * 60)
    
    # Create transforms
    train_preprocess_transform, train_augment_transform = get_train_transforms(target_size=config['target_size'])
    eval_transform = get_eval_transforms(target_size=config['target_size'])

    # Load training dataset from CarDD-TR
    print(f"\n1. Loading training data from {train_domain.upper()} CarDD-TR...")
    try:
        train_binary_dataset = CarDDDataset(
            domain=train_domain,
            train=True,  # Load from CarDD-TR
            sample_size=sample_size,
            preprocess_transform=train_preprocess_transform,
            # augment_transform=train_augment_transform,
            random_seed=config['random_state']
        )
        print(f"    Loaded {len(train_binary_dataset)} training samples")
    except RuntimeError as e:
        print(f"Error: Could not load training dataset: {e}")
        return

    # Load validation dataset from CarDD-VAL
    print(f"\n2. Loading validation data from {train_domain.upper()} CarDD-VAL...")
    # If using sample training, limit validation to 500 samples for faster testing
    val_sample_size = 500 if sample_size is not None else None
    try:
        test_binary_dataset = CarDDDataset(
            domain=train_domain,
            train=False,  # Load from CarDD-VAL
            sample_size=val_sample_size,  # 500 samples if training on subset, full set otherwise
            preprocess_transform=eval_transform,
            random_seed=config['random_state']
        )
        val_set_desc = f"{len(test_binary_dataset)} validation samples"
        if val_sample_size is not None:
            val_set_desc += f" (sampled from full validation set)"
        print(f"    Loaded {val_set_desc}")
    except RuntimeError as e:
        print(f"Error: Could not load validation dataset: {e}")
        return

    # Set indices for leakage checking (simplified)
    train_indices = list(range(len(train_binary_dataset)))
    test_indices = list(range(len(test_binary_dataset)))
    
    # ---- Verify no leakage between train/test (CarDD-TR vs CarDD-VAL) ----
    print("\nLeakage check (CarDD-TR vs CarDD-VAL):")
    print("  Using predefined dataset splits - should have no overlap by design")
    print("  CarDD-TR (training) and CarDD-VAL (validation) are separate splits")
    
    # Create data loaders
    print("\n3. Creating data loaders...")
    # Determine optimal num_workers (4 workers is usually good for most systems)
    num_workers = 4
    # Enable pin_memory for faster GPU transfer when using CUDA
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = create_dataloader(
        train_binary_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        prefetch_factor=4  # Prefetch batches for better GPU utilization
    )
    test_loader = create_dataloader(
        test_binary_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        prefetch_factor=4
    )
    print(f"   DataLoader configured: num_workers={num_workers}, pin_memory={use_pin_memory}")
    
    # Set device - prefer CUDA, then CPU (MPS has adaptive pooling issues)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Enable cuDNN benchmarking for faster training (if input size is constant)
        torch.backends.cudnn.benchmark = True
        print(f"\n4. Using device: {device}")
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   cuDNN benchmark: Enabled")
    else:
        device = torch.device('cpu')
        print(f"\n4. Using device: {device}")
        if torch.backends.mps.is_available():
            print("   Note: MPS device available but using CPU due to MPS adaptive pooling limitations")
    
    # Create model
    print(f"\n5. Creating {config['model_name']} model...")
    
    if config['model_name'] in ('vanilla'):
        input_size = config['target_size'][0] * config['target_size'][1] * 3  # 224 * 224 * 3
        model = get_model(
            model_name=config['model_name'],
            input_size=input_size,
            hidden_size=config['hidden_size'],
            dropout=config['dropout'],
            target_size=config['target_size']
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        # CNN models handle their own preprocessing
        model = get_model(
            model_name=config['model_name'],
            dropout=config['dropout'],
            hidden_size=config['hidden_size']
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print(f"\n6. Starting training for {config['num_epochs']} epochs...")
    print("=" * 60)
    print(f"Training dataset size: {len(train_binary_dataset)}")
    print(f"Validation dataset size: {len(test_binary_dataset)}")
    print()

    best_val_acc = 0.0
    best_epoch = -1
    best_state_dict = None
    train_losses = []
    train_grad_norms = []
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1_scores = []
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 40)

        # Train
        start_time = time.time()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, track_time=args.track_time)
        train_time = time.time() - start_time

        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, criterion, device)

        # Store metrics
        train_losses.append(train_metrics['loss'])
        train_grad_norms.append(train_metrics['grad_norm'])
        train_accuracies.append(train_metrics['accuracy'])
        train_precisions.append(train_metrics['precision'])
        train_recalls.append(train_metrics['recall'])
        train_f1_scores.append(train_metrics['f1_score'])
        test_losses.append(test_metrics['loss'])
        test_accuracies.append(test_metrics['accuracy'])
        test_precisions.append(test_metrics['precision'])
        test_recalls.append(test_metrics['recall'])
        test_f1_scores.append(test_metrics['f1_score'])

        # Print epoch summary
        print(f"Train Loss={train_metrics['loss']:.4f}, Grad Norm={train_metrics['grad_norm']:.4f}, Train Acc={train_metrics['accuracy']:.2f}%, "
              f"Val Loss={test_metrics['loss']:.4f}, Val Acc={test_metrics['accuracy']:.2f}%, Time={train_time:.2f}s")

        # Save best model
        if test_metrics['accuracy'] > best_val_acc:
            best_val_acc = test_metrics['accuracy']
            best_epoch = epoch + 1
            # Store a CPU copy of the current best weights to save later
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"✓ New best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Prepare results
    results = {
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'train_grad_norms': train_grad_norms,
        'train_accuracies': train_accuracies,
        'train_precisions': train_precisions,
        'train_recalls': train_recalls,
        'train_f1_scores': train_f1_scores,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'test_precisions': test_precisions,
        'test_recalls': test_recalls,
        'test_f1_scores': test_f1_scores,
        'train_samples': len(train_binary_dataset),
        'validation_samples': len(test_binary_dataset)
    }
    
    # Add domain information to config
    config['domain'] = config['train_domain']
    
    # Save model with organized structure
    model_dir = save_pytorch_model(
        model=model,
        model_name=config['model_name'],
        results=results,
        config=config
    )
    
    # If we captured a best model during training, save it alongside the final model
    try:
        if best_state_dict is not None:
            best_model_path = model_dir / f"{config['model_name']}_best.pth"
            torch.save(best_state_dict, best_model_path)
            best_ckpt_path = model_dir / f"{config['model_name']}_best_checkpoint.pth"
            torch.save({
                'model_state_dict': best_state_dict,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'config': config,
                'results': results,
            }, best_ckpt_path)
            # Update metadata.json to reference best files
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    md = json.load(f)
                md['best_model_file'] = best_model_path.name
                md['best_checkpoint_file'] = best_ckpt_path.name
                md['best_epoch'] = best_epoch
                with open(metadata_path, 'w') as f:
                    json.dump(md, f, indent=2)
            print(f"Saved best model to: {best_model_path.name} (epoch {best_epoch}, acc {best_val_acc:.2f}%)")
    except Exception as e:
        print(f"Warning: failed to save best model: {e}")
    
    # Plot and save training/validation loss curves
    try:
        plot_path = model_dir / "loss_curve.png"
        plot_losses(train_losses, val_losses=test_losses, title="Training vs Validation Loss", save_path=plot_path, show=False)
    except Exception as e:
        print(f"Warning: failed to plot loss curves: {e}")

    # Plot and save gradient norm curves
    try:
        grad_plot_path = model_dir / "gradient_norm_curve.png"
        plot_training_curves(train_grad_norms, title="Gradient Norm During Training", save_path=grad_plot_path, show=False)
    except Exception as e:
        print(f"Warning: failed to plot gradient curves: {e}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
