import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import os
import time
from pathlib import Path
from tqdm import tqdm

from utils import create_tracker, save_training_summary
from model_dann import ReverseLayerF

# ============================================
# Lambda Schedule (from DANN paper)
# ============================================
def compute_lambda_schedule(epoch, total_epochs, gamma=10.0, zeta=1.0):
    """
    Compute lambda parameter using schedule from DANN paper.

    Lambda gradually increases from 0 to zeta during training following:
        lambda_p = zeta * (2 / (1 + exp(-gamma * p)) - 1)

    where p = epoch / total_epochs (training progress)

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of training epochs
        gamma: Sharpness of the schedule (default: 10.0 from paper)
        zeta: Maximum adaptation strength in [0, 1] (default: 1.0)

    Returns:
        lambda_p: Adaptation strength in [0, zeta]
    """
    p = float(epoch) / float(total_epochs)
    lambda_p = zeta * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)
    return lambda_p

# ============================================
# Training and Evaluation Functions
# ============================================
def step_dann_epoch(model, source_loader, target_loader, optimizer,
                    loss_class, loss_domain, device, epoch, n_epoch, mode = 'train'):
    """
    Train DANN for one epoch.

    Args:
        model: CNNModel instance
        source_loader: DataLoader for source domain (MNIST)
        target_loader: DataLoader for target domain (MNIST-M)
        optimizer: Optimizer
        loss_class: Classification loss function
        loss_domain: Domain classification loss function
        device: Device (CPU/CUDA)
        epoch: Current epoch number
        n_epoch: Total number of epochs

    Returns:
        avg_src_label_loss: Average source label classification loss
        avg_tgt_label_loss: Average target label classification loss (monitoring only)
        avg_domain_loss: Average domain classification loss
        src_label_accuracy: Source label classification accuracy
        tgt_label_accuracy: Target label classification accuracy (monitoring only)
        domain_accuracy: Domain classification accuracy
    """
    if mode == 'train':
        model.train()
    else:
        model.eval()

    # Calculate lambda (adaptation strength) using schedule from DANN paper
    len_dataloader = min(len(source_loader), len(target_loader))
    total_batches = len_dataloader * n_epoch

    # Create iterators
    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_loader)

    n_batches = 0

    # Loss tracking variables
    total_src_label_loss = 0.0
    total_tgt_label_loss = 0.0
    total_domain_loss = 0.0

    # Accuracy tracking variables
    total_src_label_correct = 0
    total_src_label_samples = 0
    total_tgt_label_correct = 0
    total_tgt_label_samples = 0
    total_src_domain_correct = 0
    total_src_domain_samples = 0
    total_tgt_domain_correct = 0
    total_tgt_domain_samples = 0

    # Initialize tqdm progress bar
    pbar = tqdm(range(len_dataloader),
                desc=f'Epoch {epoch+1}/{n_epoch}',
                unit='batch',
                leave=False)

    for batch_idx in pbar:
        # Calculate progress p and lambda
        p = float(batch_idx + epoch * len_dataloader) / total_batches
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Forward pass
        model.zero_grad()

        # ============================================
        # Train on source data (MNIST)
        # ============================================
        try:
            source_data = next(data_source_iter)
        except StopIteration:
            data_source_iter = iter(source_loader)
            source_data = next(data_source_iter)

        s_img, s_label = source_data
        batch_size = len(s_label)

        # Move to device
        s_img = s_img.to(device)
        s_label = s_label.to(device)

        # Convert grayscale MNIST to RGB (3 channels)
        if s_img.shape[1] == 1:
            s_img = s_img.repeat(1, 3, 1, 1)

        src_label_output, src_domain_output = model(s_img, alpha)

        # Label loss (source domain = 0)
        err_s_label = loss_class(src_label_output, s_label)

        # Domain loss (source domain = 0)
        src_domain_label = torch.zeros(batch_size).long().to(device)
        err_s_domain = loss_domain(src_domain_output, src_domain_label)

        # ============================================
        # Train on target data (MNIST-M)
        # ============================================
        try:
            target_data = next(data_target_iter)
        except StopIteration:
            data_target_iter = iter(target_loader)
            target_data = next(data_target_iter)

        t_img, t_label = target_data
        batch_size = len(t_label)

        # Move to device
        t_img = t_img.to(device)
        t_label = t_label.to(device)

        # Forward pass
        tgt_label_output, tgt_domain_output = model(t_img, alpha)

        # Label loss (for monitoring only, not part of total loss)
        err_t_label = loss_class(tgt_label_output, t_label)

        # Domain loss (target domain = 1)
        tgt_domain_label = torch.ones(batch_size).long().to(device)
        err_t_domain = loss_domain(tgt_domain_output, tgt_domain_label)

        # Total loss
        err = err_s_label + err_s_domain + err_t_domain
        if mode == 'train':
            err.backward()
            optimizer.step()

        # Track losses
        total_src_label_loss += err_s_label.item()
        total_tgt_label_loss += err_t_label.item()
        total_domain_loss += (err_s_domain.item() + err_t_domain.item())

        # Track accuracies
        # Label accuracy (source domain)
        src_label_pred = src_label_output.data.max(1, keepdim=True)[1].squeeze()
        src_label_correct = (src_label_pred == s_label).sum().item()
        total_src_label_correct += src_label_correct
        total_src_label_samples += s_label.size(0)

        # Label accuracy (target domain)
        tgt_label_pred = tgt_label_output.data.max(1, keepdim=True)[1].squeeze()
        tgt_label_correct = (tgt_label_pred == t_label).sum().item()
        total_tgt_label_correct += tgt_label_correct
        total_tgt_label_samples += t_label.size(0)

        # Domain accuracy (source domain)
        src_domain_pred = src_domain_output.data.max(1, keepdim=True)[1].squeeze()
        src_domain_correct = (src_domain_pred == src_domain_label).sum().item()
        total_src_domain_correct += src_domain_correct
        total_src_domain_samples += batch_size

        # Domain accuracy (target domain)
        tgt_domain_pred = tgt_domain_output.data.max(1, keepdim=True)[1].squeeze()
        tgt_domain_correct = (tgt_domain_pred == tgt_domain_label).sum().item()
        total_tgt_domain_correct += tgt_domain_correct
        total_tgt_domain_samples += batch_size

        n_batches += 1

        # Update progress bar with running average losses
        running_avg_label_loss = total_src_label_loss / n_batches
        running_avg_domain_loss = total_domain_loss / n_batches

        pbar.set_postfix({
            'Class Loss': f'{running_avg_label_loss:.4f}',
            'Domain Loss': f'{running_avg_domain_loss:.4f}',
            'λ': f'{alpha:.3f}'
        })

        # Refresh display immediately
        pbar.refresh()

    # Close progress bar
    pbar.close()

    # Compute final accuracies
    src_label_accuracy = total_src_label_correct / total_src_label_samples if total_src_label_samples > 0 else 0.0
    tgt_label_accuracy = total_tgt_label_correct / total_tgt_label_samples if total_tgt_label_samples > 0 else 0.0
    domain_accuracy = (total_src_domain_correct + total_tgt_domain_correct) / (total_src_domain_samples + total_tgt_domain_samples) if (total_src_domain_samples + total_tgt_domain_samples) > 0 else 0.0

    # Return average losses and accuracies
    avg_src_label_loss = total_src_label_loss / n_batches
    avg_tgt_label_loss = total_tgt_label_loss / n_batches
    avg_domain_loss = total_domain_loss / n_batches

    return avg_src_label_loss, avg_tgt_label_loss, avg_domain_loss, src_label_accuracy, tgt_label_accuracy, domain_accuracy

# ============================================
# Main Training Function
# ============================================
def train_dann(source_name='mnist', 
            get_src_loaders = None, 
            target_name='mnist_m', 
            get_tgt_loaders = None, 
            input_model = None,
            n_epoch=100,
            batch_size=64, 
            lr=1e-3, 
            save_interval=10, 
            zeta=1.0,
            save_dir=None):
    """
    Train DANN model for MNIST → MNIST-M adaptation.

    Args:
        source_name: Source domain name ('mnist')
        target_name: Target domain name ('mnist_m')
        n_epoch: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        save_interval: Save model every N epochs
        zeta: Maximum adaptation strength (default 1.0)
    """
    print("=" * 70)
    print("DOMAIN ADVERSARIAL NEURAL NETWORK (DANN)")
    print(f"Source: {source_name.upper()} → Target: {target_name.upper()}")
    print("=" * 70)

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    if source_name == 'mnist':
        source_train_loader, source_test_loader = get_src_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported source domain: {source_name}")

    if target_name == 'mnist_m':
        target_train_loader, target_test_loader = get_tgt_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported target domain: {target_name}")

    print(f"Source train size: {len(source_train_loader.dataset)}")
    print(f"Target train size: {len(target_train_loader.dataset)}")

    # Create model
    model = input_model()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_class = nn.NLLLoss()
    loss_domain = nn.NLLLoss()

    # Move to device
    model = model.to(device)
    loss_class = loss_class.to(device)
    loss_domain = loss_domain.to(device)

    # Create training tracker (creates timestamped folder)
    tracker = create_tracker(save_dir=save_dir)
    
    # Use tracker's save directory for model checkpoints
    model_dir = tracker.save_dir

    # Training loop
    print("\nStarting training...")
    best_source_acc = 0.0
    best_target_acc = 0.0

    for epoch in range(n_epoch):
        print(f"\nEpoch {epoch+1}/{n_epoch}")
        print("-" * 50)

        epoch_start_time = time.time()

        # Train for one epoch
        src_label_loss, tgt_label_loss, domain_loss, src_label_acc, tgt_label_acc, domain_acc = step_dann_epoch(
            model, source_train_loader, target_train_loader,
            optimizer, loss_class, loss_domain, device, epoch, n_epoch
        )

        # Test for one epoch
        val_src_label_loss, val_tgt_label_loss, val_domain_loss, val_src_label_acc, val_tgt_label_acc, val_domain_acc = step_dann_epoch(
            model, source_test_loader, target_test_loader,
            optimizer, loss_class, loss_domain, device, epoch, n_epoch, mode = 'eval'
        )
        epoch_time = time.time() - epoch_start_time

        # Update tracker with epoch metrics
        tracker.update_epoch_metrics(
            train_label_loss=src_label_loss,        # Source label classification loss
            train_label_target_loss=tgt_label_loss,  # Target label classification loss (monitoring only)
            train_domain_loss=domain_loss,
            train_label_source_acc=src_label_acc,  # Training label accuracy (source only)
            train_label_target_acc=tgt_label_acc,  # In-sample target label accuracy (labels available for monitoring)
            train_domain_acc=domain_acc,       # Training domain accuracy
            val_label_source_loss=val_src_label_loss,
            val_label_target_loss=val_tgt_label_loss,
            val_label_source_acc=val_src_label_acc,
            val_label_target_acc=val_tgt_label_acc,
            val_domain_loss=val_domain_loss,  
            val_domain_accuracy=val_domain_acc,
            lambda_value=compute_lambda_schedule(epoch, n_epoch, zeta=zeta),
            epoch_time=epoch_time
        )

        # Track gradients (call after loss.backward() in train_dann_epoch)
        # Note: This would need to be modified in train_dann_epoch to track gradients
        # tracker.track_gradients(model)

        # Save best models
        if val_src_label_acc > best_source_acc:
            best_source_acc = val_src_label_acc
            torch.save(model.state_dict(), model_dir / 'best_source_model.pth')

        if val_tgt_label_acc > best_target_acc:
            best_target_acc = val_tgt_label_acc
            torch.save(model.state_dict(), model_dir / 'best_target_model.pth')

        # Save model periodically
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), model_dir / f'model_epoch_{epoch+1}.pth')

        print(f"Epoch {epoch+1}/{n_epoch} | Time: {epoch_time:.1f}s")
        print(f" | (TRAIN) Source Acc: {src_label_acc:.4f}, Loss: {src_label_loss:.4f} | Target Acc: {tgt_label_acc:.4f}, Loss: {tgt_label_loss:.4f} | Domain Loss: {domain_loss:.4f}")
        print(f" | (VAL) Source Acc: {val_src_label_acc:.4f}, Loss: {val_src_label_loss:.4f} | Target Acc: {val_tgt_label_acc:.4f}, Loss: {val_tgt_label_loss:.4f} | Domain Loss: {val_domain_loss:.4f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print(f"Best Source Accuracy: {best_source_acc:.4f}")
    print(f"Best Target Accuracy: {best_target_acc:.4f}")
    print("=" * 70)

    # Generate plots and save results
    print("\nGenerating training plots...")
    tracker.generate_all_plots()
    tracker.save_metrics()
    tracker.print_summary()

    # Save training summary
    config = {
        'source_name': source_name,
        'target_name': target_name,
        'n_epoch': n_epoch,
        'batch_size': batch_size,
        'learning_rate': lr,
        'save_interval': save_interval
    }
    save_training_summary(tracker, model, config)

    print(f"\nAll results and models saved to {tracker.save_dir}/")