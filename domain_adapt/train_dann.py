#!/usr/bin/env python3
"""
Training script for Domain Adversarial Neural Network (DANN).

This script trains a model on:
- Source domain (SD2): Labeled real + fake images
- Target domain (Kontext): Labeled real + fake images

The model learns to:
1. Classify real vs fake (main task)
2. Be invariant to which generative model created the fakes (domain adaptation)
"""

import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

# Add simple_detect_car directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "simple_detect_car"))

from data_loader import (
    CarScratchDataset,
    create_dataloader,
    get_default_transforms,
    get_eval_transforms,
)
from utils import plot_losses

# Add current directory to path for model_dann
sys.path.append(str(Path(__file__).parent))
from model_dann import get_dann_model, compute_lambda_schedule


# ============================================
# Dual-Domain Dataset Wrapper
# ============================================
class DualDomainDataset(Dataset):
    """
    Wrapper around CarScratchDataset that returns:
    (image, label, domain)

    - image: The actual image tensor
    - label: 0=real, 1=fake (for classification task)
    - domain: 0=source (SD2), 1=target (Kontext)
              Only meaningful when label=1 (fake images)
    """

    def __init__(self, data_dir, metadata_dir, domain_id, transform=None, sample_size=None):
        """
        Initialize dual-domain dataset.

        Args:
            data_dir: Path to data directory
            metadata_dir: Path to metadata directory
            domain_id: 0 for source (SD2), 1 for target (Kontext)
            transform: Optional transform to apply
            sample_size: Optional number of samples to use
        """
        self.base_dataset = CarScratchDataset.load_binary_dataset(
            data_dir=data_dir,
            metadata_dir=metadata_dir,
            sample_size=sample_size,
            shuffle=True,
            transform=transform
        )
        self.domain_id = domain_id

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Get a single item.

        Returns:
            image: Transformed image tensor
            label: 0 (real) or 1 (fake)
            domain: Domain ID (0=source, 1=target)
        """
        image, label = self.base_dataset[idx]

        # Domain ID (only meaningful for fake images where label=1)
        domain = self.domain_id

        return image, label, domain


# ============================================
# Training Functions
# ============================================
def train_dann_epoch(model, source_loader, target_loader,
                     criterion_label, criterion_domain,
                     optimizer, device, epoch, total_epochs, gamma=10.0):
    """
    Train DANN for one epoch.

    Key aspects:
    1. Label loss computed on ALL source images (real + fake)
    2. Domain loss computed ONLY on fake images (SD2-fakes vs Kontext-fakes)
    3. Lambda gradually increases from 0 to 1

    Args:
        model: DomainAdversarialNN model
        source_loader: DataLoader for source domain (SD2)
        target_loader: DataLoader for target domain (Kontext)
        criterion_label: Loss for real/fake classification
        criterion_domain: Loss for domain classification
        optimizer: Optimizer
        device: Device (CPU/CUDA/MPS)
        epoch: Current epoch number (0-indexed)
        total_epochs: Total number of epochs
        gamma: Sharpness of lambda schedule (default 10.0 from DANN paper)

    Returns:
        Dictionary with training metrics
    """
    model.train()

    # Compute lambda using schedule from DANN paper
    lambda_p = compute_lambda_schedule(epoch, total_epochs, gamma=gamma)

    # Tracking metrics
    total_label_loss = 0.0
    total_domain_loss = 0.0
    total_loss = 0.0
    label_correct = 0
    label_total = 0
    domain_correct = 0
    domain_total = 0
    num_batches = 0

    # Confusion matrix for precision/recall calculation
    tp_label = 0  # True Positive (correctly identified fake)
    tn_label = 0  # True Negative (correctly identified real)
    fp_label = 0  # False Positive (real classified as fake)
    fn_label = 0  # False Negative (fake classified as real)

    # Create iterators
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    # Iterate through the smaller of the two loaders
    num_iterations = min(len(source_loader), len(target_loader))

    # Progress bar
    pbar = tqdm(range(num_iterations), desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)

    for batch_idx in pbar:
        # ============================================
        # Get batches from both domains
        # ============================================
        try:
            src_imgs, src_labels, src_domains = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            src_imgs, src_labels, src_domains = next(source_iter)

        try:
            tgt_imgs, tgt_labels, tgt_domains = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            tgt_imgs, tgt_labels, tgt_domains = next(target_iter)

        # Move to device
        src_imgs = src_imgs.to(device)
        src_labels = src_labels.to(device)

        tgt_imgs = tgt_imgs.to(device)
        tgt_labels = tgt_labels.to(device)

        # Derive is_fake from labels (is_fake = label == 1)
        src_is_fake = (src_labels == 1)
        tgt_is_fake = (tgt_labels == 1)

        # ============================================
        # Forward pass
        # ============================================
        src_label_pred, src_domain_pred = model(src_imgs, alpha=lambda_p)
        tgt_label_pred, tgt_domain_pred = model(tgt_imgs, alpha=lambda_p)

        # ============================================
        # Loss 1: Label Classification (Real vs Fake)
        # ============================================
        # Computed on source domain images (we have labels)
        # In practice, could also use target labels if available (semi-supervised)
        label_loss = criterion_label(src_label_pred.squeeze(), src_labels.float())

        # ============================================
        # Loss 2: Domain Classification (SD2-fake vs Kontext-fake)
        # ============================================
        # ONLY computed on fake images from both domains
        domain_loss = torch.tensor(0.0, device=device)

        # Source domain: Extract fake images (SD2-generated fakes)
        src_fake_mask = src_is_fake.bool()
        if src_fake_mask.sum() > 0:
            src_fake_domain_pred = src_domain_pred[src_fake_mask]
            src_fake_domain_labels = torch.zeros(src_fake_mask.sum(), device=device)  # 0 = SD2
            domain_loss = domain_loss + criterion_domain(
                src_fake_domain_pred.squeeze(),
                src_fake_domain_labels
            )
            # Track accuracy
            src_domain_pred_binary = (src_fake_domain_pred.squeeze() > 0.5).float()
            domain_correct += (src_domain_pred_binary == src_fake_domain_labels).sum().item()
            domain_total += src_fake_mask.sum().item()

        # Target domain: Extract fake images (Kontext-generated fakes)
        tgt_fake_mask = tgt_is_fake.bool()
        if tgt_fake_mask.sum() > 0:
            tgt_fake_domain_pred = tgt_domain_pred[tgt_fake_mask]
            tgt_fake_domain_labels = torch.ones(tgt_fake_mask.sum(), device=device)  # 1 = Kontext
            domain_loss = domain_loss + criterion_domain(
                tgt_fake_domain_pred.squeeze(),
                tgt_fake_domain_labels
            )
            # Track accuracy
            tgt_domain_pred_binary = (tgt_fake_domain_pred.squeeze() > 0.5).float()
            domain_correct += (tgt_domain_pred_binary == tgt_fake_domain_labels).sum().item()
            domain_total += tgt_fake_mask.sum().item()

        # ============================================
        # Total Loss
        # ============================================
        total_batch_loss = label_loss + domain_loss

        # ============================================
        # Backward pass and optimization
        # ============================================
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        # ============================================
        # Track metrics
        # ============================================
        total_label_loss += label_loss.item()
        total_domain_loss += domain_loss.item()
        total_loss += total_batch_loss.item()

        # Label prediction accuracy (on source)
        src_label_pred_binary = (src_label_pred.squeeze() > 0.5).float()
        label_correct += (src_label_pred_binary == src_labels).sum().item()
        label_total += src_labels.size(0)

        # Update confusion matrix for precision/recall calculation
        for pred, true in zip(src_label_pred_binary, src_labels):
            if true == 0:  # Real image
                if pred == 0:
                    tn_label += 1
                else:
                    fp_label += 1
            else:  # Fake image
                if pred == 1:
                    tp_label += 1
                else:
                    fn_label += 1

        num_batches += 1

        # Update progress bar with current metrics
        if num_batches > 0:
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'acc': f'{100.0*label_correct/max(1,label_total):.1f}%',
                'λ': f'{lambda_p:.3f}'
            })

    # Compute average metrics
    avg_label_loss = total_label_loss / num_batches
    avg_domain_loss = total_domain_loss / num_batches
    avg_total_loss = total_loss / num_batches
    label_accuracy = 100.0 * label_correct / max(1, label_total)
    domain_accuracy = 100.0 * domain_correct / max(1, domain_total)

    # Calculate precision and recall
    precision = tp_label / max(1, tp_label + fp_label)
    recall = tp_label / max(1, tp_label + fn_label)
    f1_score = 2 * (precision * recall) / max(1e-8, precision + recall)

    return {
        'label_loss': avg_label_loss,
        'domain_loss': avg_domain_loss,
        'total_loss': avg_total_loss,
        'label_accuracy': label_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'domain_accuracy': domain_accuracy,
        'lambda': lambda_p,
        'confusion_matrix': {
            'tp': tp_label,
            'tn': tn_label,
            'fp': fp_label,
            'fn': fn_label
        }
    }


def evaluate_dann(model, data_loader, device, domain_name="test"):
    """
    Evaluate DANN model.

    Tracks metrics separately for:
    - Real images
    - Fake images
    - Domain classification (on fakes only)

    Args:
        model: DomainAdversarialNN model
        data_loader: DataLoader for evaluation
        device: Device (CPU/CUDA/MPS)
        domain_name: Name for logging

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    criterion_label = nn.BCELoss()

    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    domain_correct = 0
    domain_total = 0
    total_loss = 0.0

    # Confusion matrix for label classification
    tp_label = 0  # True Positive (correctly identified fake)
    tn_label = 0  # True Negative (correctly identified real)
    fp_label = 0  # False Positive (real classified as fake)
    fn_label = 0  # False Negative (fake classified as real)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Evaluating {domain_name}", leave=False)
        for images, labels, domains in pbar:
            images = images.to(device)
            labels = labels.to(device)
            domains = domains.to(device)

            # Derive is_fake from labels
            is_fake = (labels == 1)

            # Forward pass (alpha=0 for inference)
            label_pred, domain_pred = model(images, alpha=0.0)

            # Loss
            loss = criterion_label(label_pred.squeeze(), labels.float())
            total_loss += loss.item()

            # Label predictions
            predicted_labels = (label_pred.squeeze() > 0.5).float()

            # Separate metrics for real and fake images
            real_mask = ~is_fake.bool()
            fake_mask = is_fake.bool()

            if real_mask.sum() > 0:
                real_correct += (predicted_labels[real_mask] == labels[real_mask]).sum().item()
                real_total += real_mask.sum().item()

            if fake_mask.sum() > 0:
                fake_correct += (predicted_labels[fake_mask] == labels[fake_mask]).sum().item()
                fake_total += fake_mask.sum().item()

            # Update confusion matrix for precision/recall calculation
            for pred, true in zip(predicted_labels, labels):
                if true == 0:  # Real image
                    if pred == 0:
                        tn_label += 1
                    else:
                        fp_label += 1
                else:  # Fake image
                    if pred == 1:
                        tp_label += 1
                    else:
                        fn_label += 1

            # Domain classification accuracy (only on fakes)
            predicted_domains = (domain_pred[fake_mask].squeeze() > 0.5).float()
            domain_correct += (predicted_domains == domains[fake_mask].float()).sum().item()
            domain_total += fake_mask.sum().item()

    avg_loss = total_loss / len(data_loader)
    real_acc = 100.0 * real_correct / max(1, real_total)
    fake_acc = 100.0 * fake_correct / max(1, fake_total)
    overall_acc = 100.0 * (real_correct + fake_correct) / max(1, real_total + fake_total)
    domain_acc = 100.0 * domain_correct / max(1, domain_total)

    # Calculate precision and recall
    precision = tp_label / max(1, tp_label + fp_label)
    recall = tp_label / max(1, tp_label + fn_label)
    f1_score = 2 * (precision * recall) / max(1e-8, precision + recall)

    return {
        'loss': avg_loss,
        'real_accuracy': real_acc,
        'fake_accuracy': fake_acc,
        'overall_accuracy': overall_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'domain_accuracy': domain_acc,
        'real_total': real_total,
        'fake_total': fake_total,
        'confusion_matrix': {
            'tp': tp_label,
            'tn': tn_label,
            'fp': fp_label,
            'fn': fn_label
        }
    }


# ============================================
# Main Training Script
# ============================================
def main():
    print("=" * 70)
    print("DOMAIN ADVERSARIAL NEURAL NETWORK (DANN) TRAINING")
    print("=" * 70)

    # ============================================
    # Configuration
    # ============================================
    config = {
        # Source domain (SD2)
        'source_name': 'SD2',
        'source_data_type': 'CarDD-TR',
        'source_data_dir': None,  # Will be set below
        'source_metadata_dir': None,

        # Target domain (Kontext)
        'target_name': 'Kontext',
        'target_data_type': 'CarDD-TR',  # Use same split or different?
        'target_data_dir': None,
        'target_metadata_dir': None,

        # Training parameters
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 1e-3,
        'target_size': (512, 512),  # Reduced from 512 for speed (4x faster!)
        'sample_size': 200,  # Use None for full dataset

        # Model architecture
        'input_channels': 3,
        'num_classes': 1,
        'feature_hidden_size': 256,
        'domain_hidden_size': 64,
        'dropout': 0.2,

        # Lambda schedule
        'gamma': 10.0,  # Sharpness of lambda schedule (default 10.0 from DANN paper)

        # Other
        'random_state': 42
    }

    # Set up paths (relative to this script's location)
    script_dir = Path(__file__).parent
    base_path = script_dir.parent / "cardd_data" / "GenAI_Results"

    config['source_data_dir'] = str(base_path / config['source_name'] / config['source_data_type'])
    config['source_metadata_dir'] = str(base_path / config['source_name'] / config['source_data_type'] / "metadata")

    config['target_data_dir'] = str(base_path / config['target_name'] / config['target_data_type'])
    config['target_metadata_dir'] = str(base_path / config['target_name'] / config['target_data_type'] / "metadata")

    print(f"\nConfiguration:")
    print(f"  Source: {config['source_name']}/{config['source_data_type']}")
    print(f"  Target: {config['target_name']}/{config['target_data_type']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")

    # ============================================
    # Create datasets and dataloaders
    # ============================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_transform = get_default_transforms(target_size=config['target_size'], augment=True)
    eval_transform = get_eval_transforms(target_size=config['target_size'])

    # Source domain dataset (SD2)
    print(f"\nLoading source domain ({config['source_name']})...")
    source_dataset = DualDomainDataset(
        data_dir=config['source_data_dir'],
        metadata_dir=config['source_metadata_dir'],
        domain_id=0,  # Source = 0
        transform=train_transform,
        sample_size=config['sample_size']
    )
    print(f"  Source dataset size: {len(source_dataset)}")

    # Target domain dataset (Kontext)
    print(f"\nLoading target domain ({config['target_name']})...")
    target_dataset = DualDomainDataset(
        data_dir=config['target_data_dir'],
        metadata_dir=config['target_metadata_dir'],
        domain_id=1,  # Target = 1
        transform=train_transform,
        sample_size=config['sample_size']
    )
    print(f"  Target dataset size: {len(target_dataset)}")

    # Create dataloaders
    source_loader = create_dataloader(
        source_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        target_size=config['target_size']
    )

    target_loader = create_dataloader(
        target_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        target_size=config['target_size']
    )

    # Create evaluation dataset (target domain with eval transform)
    target_eval_dataset = DualDomainDataset(
        data_dir=config['target_data_dir'],
        metadata_dir=config['target_metadata_dir'],
        domain_id=1,
        transform=eval_transform,
        sample_size=config['sample_size']
    )

    target_eval_loader = create_dataloader(
        target_eval_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        target_size=config['target_size']
    )

    # ============================================
    # Create model
    # ============================================
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = get_dann_model(
        input_channels=config['input_channels'],
        num_classes=config['num_classes'],
        feature_hidden_size=config['feature_hidden_size'],
        domain_hidden_size=config['domain_hidden_size'],
        dropout=config['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions
    criterion_label = nn.BCELoss()
    criterion_domain = nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # ============================================
    # Training loop
    # ============================================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_target_acc = 0.0
    best_epoch = -1
    best_state_dict = None

    # History tracking
    history = {
        'train_label_loss': [],
        'train_domain_loss': [],
        'train_total_loss': [],
        'train_label_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1_score': [],
        'train_domain_accuracy': [],
        'target_eval_loss': [],
        'target_eval_accuracy': [],
        'target_eval_real_accuracy': [],
        'target_eval_fake_accuracy': [],
        'target_eval_precision': [],
        'target_eval_recall': [],
        'target_eval_f1_score': [],
        'target_eval_domain_accuracy': [],
        'lambda_values': [],
        'epoch_train_times': [],
        'epoch_eval_times': []
    }

    # Track total training time
    from datetime import datetime
    training_start_time = time.time()
    training_start_datetime = datetime.now()
    print(f"\nTraining started at: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 70)

        # Train
        epoch_start_time = time.time()
        train_metrics = train_dann_epoch(
            model, source_loader, target_loader,
            criterion_label, criterion_domain,
            optimizer, device, epoch, config['num_epochs'], config['gamma']
        )
        train_time = time.time() - epoch_start_time

        # Evaluate on target domain
        eval_start_time = time.time()
        target_metrics = evaluate_dann(model, target_eval_loader, device, domain_name="target")
        eval_time = time.time() - eval_start_time

        # Print metrics in symmetric table format
        print(f"")
        print(f"{'─'*70}")
        print(f"{'Metric':<20} {'Train':>12} {'Target':>12} {'Info':>22}")
        print(f"{'─'*70}")
        print(f"{'Label Loss':<20} {train_metrics['label_loss']:>12.4f} {target_metrics['loss']:>12.4f}")
        print(f"{'Domain Loss':<20} {train_metrics['domain_loss']:>12.4f} {'N/A':>12} {'(adversarial)':>12}")
        print(f"{'Total Loss':<20} {train_metrics['total_loss']:>12.4f} {'N/A':>12}")
        print(f"{'Label Accuracy':<20} {train_metrics['label_accuracy']:>11.2f}% {target_metrics['overall_accuracy']:>11.2f}%")
        print(f"{'Precision':<20} {train_metrics['precision']:>12.4f} {target_metrics['precision']:>12.4f}")
        print(f"{'Recall':<20} {train_metrics['recall']:>12.4f} {target_metrics['recall']:>12.4f}")
        print(f"{'F1 Score':<20} {train_metrics['f1_score']:>12.4f} {target_metrics['f1_score']:>12.4f}")
        print(f"{'─'*70}")

        # Domain accuracy (combined - only on fake images)
        avg_domain_acc = (train_metrics['domain_accuracy'] + target_metrics['domain_accuracy']) / 2
        domain_status = "✓ Good" if 45 <= avg_domain_acc <= 55 else "⚠ Check"
        print(f"{'Domain Accuracy':<20} {avg_domain_acc:>11.2f}% {'(fakes only)':>12} {domain_status:>10}")

        print(f"{'─'*70}")
        print(f"Lambda: {train_metrics['lambda']:.4f} | Train: {train_time:.1f}s | Eval: {eval_time:.1f}s | Total: {train_time+eval_time:.1f}s")

        # Store history
        history['train_label_loss'].append(train_metrics['label_loss'])
        history['train_domain_loss'].append(train_metrics['domain_loss'])
        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_label_accuracy'].append(train_metrics['label_accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['train_f1_score'].append(train_metrics['f1_score'])
        history['train_domain_accuracy'].append(train_metrics['domain_accuracy'])
        history['target_eval_loss'].append(target_metrics['loss'])
        history['target_eval_accuracy'].append(target_metrics['overall_accuracy'])
        history['target_eval_real_accuracy'].append(target_metrics['real_accuracy'])
        history['target_eval_fake_accuracy'].append(target_metrics['fake_accuracy'])
        history['target_eval_precision'].append(target_metrics['precision'])
        history['target_eval_recall'].append(target_metrics['recall'])
        history['target_eval_f1_score'].append(target_metrics['f1_score'])
        history['target_eval_domain_accuracy'].append(target_metrics['domain_accuracy'])
        history['lambda_values'].append(train_metrics['lambda'])
        history['epoch_train_times'].append(train_time)
        history['epoch_eval_times'].append(eval_time)

        # Save best model
        if target_metrics['overall_accuracy'] > best_target_acc:
            best_target_acc = target_metrics['overall_accuracy']
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"🏆 New best target accuracy: {best_target_acc:.2f}%")

    # Calculate total training time
    training_end_time = time.time()
    training_end_datetime = datetime.now()
    total_training_time = training_end_time - training_start_time

    print(f"\n{'='*70}")
    print(f"Training completed at: {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    print(f"Average time per epoch: {total_training_time/config['num_epochs']:.1f}s")
    print(f"{'='*70}")

    # ============================================
    # Save model
    # ============================================
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Create timestamped directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = models_dir / f"model_dann_{timestamp}"
    model_dir.mkdir(exist_ok=True)

    # Save best model
    if best_state_dict is not None:
        best_model_path = model_dir / "dann_best.pth"
        torch.save(best_state_dict, best_model_path)
        print(f"Saved best model: {best_model_path}")

    # Save final model
    final_model_path = model_dir / "dann_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    # Save metadata
    metadata = {
        'config': config,
        'best_epoch': best_epoch,
        'best_target_accuracy': best_target_acc,
        'history': history,
        'timestamp': timestamp,
        'timing': {
            'training_start': training_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'training_end': training_end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'total_training_time_seconds': total_training_time,
            'total_training_time_minutes': total_training_time / 60,
            'total_training_time_hours': total_training_time / 3600,
            'average_epoch_time_seconds': total_training_time / config['num_epochs'],
            'average_train_time_per_epoch': sum(history['epoch_train_times']) / len(history['epoch_train_times']) if history['epoch_train_times'] else 0,
            'average_eval_time_per_epoch': sum(history['epoch_eval_times']) / len(history['epoch_eval_times']) if history['epoch_eval_times'] else 0
        }
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    # Plot training curves
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        axes[0, 0].plot(history['train_label_loss'], label='Label Loss')
        axes[0, 0].plot(history['train_domain_loss'], label='Domain Loss')
        axes[0, 0].plot(history['train_total_loss'], label='Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Classification metrics curves
        axes[0, 1].plot(history['train_label_accuracy'], label='Train Label Acc')
        axes[0, 1].plot(history['target_eval_accuracy'], label='Target Overall Acc')
        axes[0, 1].plot(history['train_precision'], label='Train Precision')
        axes[0, 1].plot(history['train_recall'], label='Train Recall')
        axes[0, 1].plot(history['train_f1_score'], label='Train F1 Score')
        axes[0, 1].plot(history['target_eval_precision'], label='Target Precision')
        axes[0, 1].plot(history['target_eval_recall'], label='Target Recall')
        axes[0, 1].plot(history['target_eval_f1_score'], label='Target F1 Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Classification Metrics')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Domain accuracy
        axes[1, 0].plot(history['train_domain_accuracy'], label='Train Domain Acc')
        axes[1, 0].plot(history['target_eval_domain_accuracy'], label='Target Domain Acc')
        axes[1, 0].axhline(y=50, color='r', linestyle='--', label='Random (50%)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Domain Accuracy (%)')
        axes[1, 0].set_title('Domain Classification Accuracy (should → 50%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Lambda schedule
        axes[1, 1].plot(history['lambda_values'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Lambda')
        axes[1, 1].set_title('Lambda Schedule')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = model_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training curves: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot training curves: {e}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best target accuracy: {best_target_acc:.2f}% at epoch {best_epoch}")
    print(f"Model saved to: {model_dir}")


if __name__ == "__main__":
    main()
