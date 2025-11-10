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
    combine_datasets,
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
    
    # Choose training domains
    print("\nSelect training domains:")
    print("  [1] SD2 only")
    print("  [2] Kontext only")
    print("  [3] SD2 + Kontext (combined)")
    domain_choice = input("Enter choice [1-3] (default: 1): ").strip() or "1"
    
    if domain_choice == "1":
        train_domains = ["SD2"]
    elif domain_choice == "2":
        train_domains = ["Kontext"]
    elif domain_choice == "3":
        train_domains = ["SD2", "Kontext"]
    else:
        print("Invalid choice, defaulting to SD2 only")
        train_domains = ["SD2"]
    
    print(f"\nTraining on: {', '.join(train_domains)}")
    
    # Choose validation strategy
    print("\nSelect validation strategy:")
    print("  [1] Use CarDD-VAL split (recommended)")
    print("  [2] Random train/test split (80/20)")
    val_choice = input("Enter choice [1-2] (default: 1): ").strip() or "1"
    use_val_split = (val_choice == "1")
    
    # Set up paths (relative to this script's location)
    script_dir = Path(__file__).parent
    genai_root = script_dir.parent / "cardd_data" / "GenAI_Results"
    
    train_data_type = "CarDD-TR"
    val_data_type = "CarDD-VAL" if use_val_split else None

    # Choose sampling strategy
    sample_size_input = input("\nSample size (number of samples, or press Enter for None/full dataset): ").strip()
    try:
        sample_size = int(sample_size_input) if sample_size_input else None
    except ValueError:
        print("Invalid sample size, using None (full dataset)")
        sample_size = None
    
    if sample_size is not None and len(train_domains) > 1:
        print("\nSampling strategy for multiple domains:")
        print("  [1] Sample from each domain separately (e.g., 500 from SD2 + 500 from Kontext)")
        print("  [2] Load all data, combine, then sample from combined (e.g., 500 total from combined SD2+Kontext)")
        sampling_strategy = input("Enter choice [1-2] (default: 1): ").strip() or "1"
        sample_per_domain = (sampling_strategy == "1")
    else:
        sample_per_domain = True  # Default behavior when single domain or no sampling
    
    # Configuration
    config = {
        'train_domains': train_domains,
        'use_val_split': use_val_split,
        'sample_size': sample_size,
        'sample_per_domain': sample_per_domain if len(train_domains) > 1 else True,
        'shuffle': True,
        'target_size': (512, 512),
        'batch_size': 32,
        'test_size': 0.2,
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'model_name': 'cnn',  # 'vanilla', 'cnn'
        'hidden_size': 256,
        'dropout': 0.2,
        'random_state': 42
    }
    
    print(f"\nConfiguration: {config}")
    
    # Create transforms
    train_transform = get_default_transforms(target_size=config['target_size'], augment=True)
    eval_transform = get_eval_transforms(target_size=config['target_size'])

    # Load training datasets from selected domains
    print("\n1. Loading training datasets...")
    train_datasets = []
    for domain in train_domains:
        train_dir = genai_root / domain / train_data_type
        train_data_dir = str(train_dir)
        train_metadata_dir = str(train_dir / "metadata")
        
        if not Path(train_data_dir).exists():
            print(f"Warning: Training directory {train_data_dir} does not exist, skipping {domain}")
            continue
        
        print(f"  Loading {domain}...")
        # If sampling per domain, use sample_size; otherwise load all and sample after combining
        domain_sample_size = config['sample_size'] if sample_per_domain else None
        ds = CarScratchDataset.load_binary_dataset(
            data_dir=train_data_dir,
            metadata_dir=train_metadata_dir,
            sample_size=domain_sample_size,
            shuffle=config['shuffle'],
            transform=train_transform,
        )
        train_datasets.append(ds)
        print(f"    Loaded {len(ds)} samples from {domain}")
    
    if not train_datasets:
        print("Error: No training datasets loaded!")
        return
    
    # Combine datasets if multiple domains
    if len(train_datasets) > 1:
        print(f"\nCombining {len(train_datasets)} datasets...")
        # If not sampling per domain, sample from combined dataset
        combined_sample_size = None if sample_per_domain else config['sample_size']
        binary_dataset = combine_datasets(train_datasets, sample_size=combined_sample_size, random_seed=config['random_state'])
        print(f"Combined training dataset size: {len(binary_dataset)}")
    else:
        binary_dataset = train_datasets[0]
    
    # Load validation/test dataset
    if use_val_split:
        print("\n2. Loading validation dataset from CarDD-VAL...")
        val_datasets = []
        for domain in train_domains:
            val_dir = genai_root / domain / val_data_type
            val_data_dir = str(val_dir)
            val_metadata_dir = str(val_dir / "metadata")
            
            if not Path(val_data_dir).exists():
                print(f"Warning: Validation directory {val_data_dir} does not exist, skipping {domain}")
                continue
            
            print(f"  Loading {domain} validation...")
            ds = CarScratchDataset.load_binary_dataset(
                data_dir=val_data_dir,
                metadata_dir=val_metadata_dir,
                sample_size=None,  # Use full validation set
                shuffle=False,
                transform=eval_transform,
            )
            val_datasets.append(ds)
            print(f"    Loaded {len(ds)} samples from {domain} validation")
        
        if not val_datasets:
            print("Warning: No validation datasets found, falling back to random split")
            use_val_split = False
        
        if val_datasets:
            if len(val_datasets) > 1:
                print(f"\nCombining {len(val_datasets)} validation datasets...")
                val_binary_dataset = combine_datasets(val_datasets)
                print(f"Combined validation dataset size: {len(val_binary_dataset)}")
            else:
                val_binary_dataset = val_datasets[0]
            
            # Create train and test datasets (no split needed when using VAL)
            train_binary_dataset = binary_dataset
            test_binary_dataset = val_binary_dataset
            train_indices = list(range(len(train_binary_dataset)))
            test_indices = list(range(len(test_binary_dataset)))
        else:
            use_val_split = False
    
    if not use_val_split:
        # Create train/test split
        print("\n2. Creating train/test split...")
        train_indices, test_indices = create_train_test_split(
            binary_dataset, 
            test_size=config['test_size'],
            random_state=config['random_state']
        )
        
        # Create train and test datasets by shallow-copying
        train_binary_dataset = copy.copy(binary_dataset)
        test_binary_dataset = copy.copy(binary_dataset)
        
        # Apply transforms per split
        train_binary_dataset.transform = train_transform
        test_binary_dataset.transform = eval_transform
        
        # Bind each to its split
        train_binary_dataset.valid_entries = [binary_dataset.valid_entries[i] for i in train_indices]
        test_binary_dataset.valid_entries = [binary_dataset.valid_entries[i] for i in test_indices]
        
        # Rebuild shuffled indices
        train_binary_dataset.shuffled_indices = list(range(len(train_binary_dataset.valid_entries)))
        test_binary_dataset.shuffled_indices = list(range(len(test_binary_dataset.valid_entries)))
    
    # ---- Verify no leakage between train/test based on image ids and file paths ----
    def _paths_for_indices(ds, indices):
        ids = set()
        orig_paths = set()
        proc_paths = set()
        valid_len = len(ds.valid_entries)
        for i in indices:
            if i >= valid_len:
                continue  # Skip out-of-bounds indices
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

    # Use valid_entries length to ensure indices are in bounds
    train_indices_list = train_indices if not use_val_split else list(range(len(train_binary_dataset.valid_entries)))
    test_indices_list = test_indices if not use_val_split else list(range(len(test_binary_dataset.valid_entries)))
    
    tr_ids, tr_orig, tr_proc = _paths_for_indices(train_binary_dataset, train_indices_list)
    te_ids, te_orig, te_proc = _paths_for_indices(test_binary_dataset, test_indices_list)

    id_overlap = tr_ids & te_ids
    orig_overlap = tr_orig & te_orig
    proc_overlap = tr_proc & te_proc
    any_overlap = bool(id_overlap or orig_overlap or proc_overlap)
    print("\nLeakage check (train vs validation/test):")
    print(f"  Image ID overlap: {len(id_overlap)}")
    print(f"  Original path overlap: {len(orig_overlap)}")
    print(f"  Processed path overlap: {len(proc_overlap)}")
    if any_overlap:
        print("WARNING: Potential data leakage detected (overlapping items between train and validation/test).")
    else:
        print("No overlap detected between train and validation/test splits (by IDs and file paths).")
    
    # Pass forbidden paths to test dataset to block leakage when sampling (only for random split)
    if not use_val_split:
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
    
    # Add information about combined dataset usage to config
    config['use_combined_dataset'] = len(config['train_domains']) > 1
    if config['use_combined_dataset']:
        config['combined_domains'] = config['train_domains']
        config['num_domains_combined'] = len(config['train_domains'])
    
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
