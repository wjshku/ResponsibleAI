#!/usr/bin/env python3
"""
Training script for neural network models (MLP, CNN).
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import copy

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import (
    CarScratchDataset,
    create_dataloader,
    create_train_test_split,
    get_default_transforms,
    get_eval_transforms,
)
from models import get_model
from utils import save_pytorch_model, plot_losses



def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
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
        
        # Backward pass
        loss.backward()
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
    
    epoch_loss = running_loss / len(train_loader)
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


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
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
    print("=" * 60)
    print("CAR IMAGE BINARY CLASSIFICATION - NEURAL NETWORKS")
    print("=" * 60)
    
    img2img_type, data_type = "SD2", "CarDD-TR"

    # Set up paths (relative to this script's location)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent / "cardd_data" / "GenAI_Results" / img2img_type / data_type
    data_dir = str(base_dir)
    metadata_dir = str(base_dir / "metadata")

    # Configuration
    config = {
        'data_dir': Path(data_dir),
        'metadata_dir': Path(metadata_dir),
        'sample_size': 500,  # Use None for full dataset
        'shuffle': True,
        'target_size': (512, 512),
        'batch_size': 32,
        'test_size': 0.2,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'model_name': 'cnn',  # 'vanilla', 'cnn'
        'hidden_size': 64,
        'dropout': 0.2,
        'random_state': 42
    }
    
    print(f"Configuration: {config}")
    
    # Check if data directory exists
    if not config['data_dir'].exists():
        print(f"Error: Data directory {config['data_dir']} does not exist!")
        return
    
    # Create transforms
    train_transform = get_default_transforms(target_size=config['target_size'], augment=True)
    eval_transform = get_eval_transforms(target_size=config['target_size'])

    # Create binary classification dataset (for computing split)
    print("\n1. Creating binary classification dataset...")
    binary_dataset = CarScratchDataset.load_binary_dataset(
        data_dir=config['data_dir'],
        metadata_dir=config['metadata_dir'],
        sample_size=config['sample_size'],
        shuffle=config['shuffle'],
    )
    
    # Create train/test split
    print("\n2. Creating train/test split...")
    train_indices, test_indices = create_train_test_split(
        binary_dataset, 
        test_size=config['test_size'],
        random_state=config['random_state']
    )

    # ---- Verify no leakage between train/test based on image ids and file paths ----
    def _paths_for_indices(ds, indices):
        ids = set()
        orig_paths = set()
        proc_paths = set()
        for i in indices:
            e = ds.valid_entries[i]
            img_id = e.get('image_id')
            if img_id is not None:
                ids.add(str(img_id))
            op = e.get('original_image_path')
            pp = e.get('processed_image_path')
            if op:
                orig_paths.add(str(op))
            if pp:
                proc_paths.add(str(pp))
        return ids, orig_paths, proc_paths

    tr_ids, tr_orig, tr_proc = _paths_for_indices(binary_dataset, train_indices)
    te_ids, te_orig, te_proc = _paths_for_indices(binary_dataset, test_indices)

    id_overlap = tr_ids & te_ids
    orig_overlap = tr_orig & te_orig
    proc_overlap = tr_proc & te_proc
    any_overlap = bool(id_overlap or orig_overlap or proc_overlap)
    print("\nLeakage check (train vs test):")
    print(f"  Image ID overlap: {len(id_overlap)}")
    print(f"  Original path overlap: {len(orig_overlap)}")
    print(f"  Processed path overlap: {len(proc_overlap)}")
    if any_overlap:
        print("WARNING: Potential data leakage detected (overlapping items between train and test).")
    else:
        print("No overlap detected between train and test splits (by IDs and file paths).")
    
    # Create train and test datasets by shallow-copying the binary_dataset to avoid re-sampling
    train_binary_dataset = copy.copy(binary_dataset)
    test_binary_dataset = copy.copy(binary_dataset)
    
    # Apply transforms per split
    train_binary_dataset.transform = train_transform
    test_binary_dataset.transform = eval_transform
    
    # Bind each to its split using the precomputed indices (sum equals original sample size)
    train_binary_dataset.valid_entries = [binary_dataset.valid_entries[i] for i in train_indices]
    test_binary_dataset.valid_entries = [binary_dataset.valid_entries[i] for i in test_indices]
    
    # Rebuild shuffled indices consistent with new lengths
    train_binary_dataset.shuffled_indices = list(range(len(train_binary_dataset.valid_entries)))
    test_binary_dataset.shuffled_indices = list(range(len(test_binary_dataset.valid_entries)))

    # Pass forbidden paths to test dataset to block leakage when sampling
    # Ensure the loader's dataset has attributes for forbidden sets
    setattr(test_binary_dataset, 'forbidden_original_paths', tr_orig)
    setattr(test_binary_dataset, 'forbidden_processed_paths', tr_proc)
    
    # Create data loaders
    print("\n3. Creating data loaders...")
    train_loader = create_dataloader(
        train_binary_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        target_size=config['target_size']
    )
    test_loader = create_dataloader(
        test_binary_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        target_size=config['target_size']
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n4. Using device: {device}")
    
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
    
    best_test_acc = 0.0
    best_epoch = -1
    best_state_dict = None
    train_losses = []
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
        # Report dataset sizes for this epoch
        try:
            print(f"Train dataset size: {len(train_binary_dataset)} | Test dataset size: {len(test_binary_dataset)}")
        except Exception:
            pass
        
        # Train
        start_time = time.time()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.time() - start_time
        
        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_metrics['loss'])
        train_accuracies.append(train_metrics['accuracy'])
        train_precisions.append(train_metrics['precision'])
        train_recalls.append(train_metrics['recall'])
        train_f1_scores.append(train_metrics['f1_score'])
        test_losses.append(test_metrics['loss'])
        test_accuracies.append(test_metrics['accuracy'])
        test_precisions.append(test_metrics['precision'])
        test_recalls.append(test_metrics['recall'])
        test_f1_scores.append(test_metrics['f1_score'])
        
        # Print results
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1_score']:.4f}")
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%, Prec: {test_metrics['precision']:.4f}, Rec: {test_metrics['recall']:.4f}, F1: {test_metrics['f1_score']:.4f}")
        print(f"Time: {train_time:.2f}s")
        
        # Save best model
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            best_epoch = epoch + 1
            # Store a CPU copy of the current best weights to save later
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"New best test accuracy: {best_test_acc:.2f}% at epoch {best_epoch}")
    
    # Prepare results
    results = {
        'best_test_acc': best_test_acc,
        'train_losses': train_losses,
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
        'test_samples': len(test_binary_dataset)
    }
    
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
                'best_test_acc': best_test_acc,
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
            print(f"Saved best model to: {best_model_path.name} (epoch {best_epoch}, acc {best_test_acc:.2f}%)")
    except Exception as e:
        print(f"Warning: failed to save best model: {e}")
    
    # Plot and save training/validation loss curves
    try:
        plot_path = model_dir / "loss_curve.png"
        plot_losses(train_losses, val_losses=test_losses, title="Training vs Validation Loss", save_path=plot_path, show=False)
    except Exception as e:
        print(f"Warning: failed to plot loss curves: {e}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
