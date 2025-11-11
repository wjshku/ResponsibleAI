#!/usr/bin/env python3
"""
Domain Adversarial Neural Network (DANN) for MNIST → MNIST-M adaptation.

This replicates the classic experiment from "Unsupervised Domain Adaptation by Backpropagation"
(Ganin & Lempitsky, 2015) where MNIST digits are adapted to the MNIST-M dataset.

The model learns to:
1. Classify MNIST digits (0-9) on source domain
2. Become domain-invariant using gradient reversal
3. Maintain classification accuracy on target domain (MNIST-M)

Architecture:
    Input Image (28x28x3)
        ↓
    Feature Extractor (CNN)
        ↓
    Features (800-dim)
        ↓
        ├→ Label Predictor → Digit classification (0-9)
        │
        └→ [GRL] → Domain Classifier → MNIST/MNIST-M classification
"""

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

# Import training utilities
from utils import create_tracker, save_training_summary


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
# Gradient Reversal Layer (GRL)
# ============================================
class ReverseLayerF(torch.autograd.Function):
    """
    Gradient Reversal Layer from the DANN paper.

    Forward pass: Identity function
    Backward pass: Multiply gradient by -alpha (reversal)
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# ============================================
# DANN Model (CNN for MNIST)
# ============================================
class CNNModel(nn.Module):
    """
    CNN Model for DANN with MNIST/MNIST-M adaptation.

    Based on the architecture from the original DANN paper.
    Adapted for 28x28 RGB images (MNIST-M has color).
    """

    def __init__(self):
        super(CNNModel, self).__init__()

        # ============================================
        # Feature Extractor (Shared CNN backbone)
        # ============================================
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        # ============================================
        # Label Predictor: Digit classification (0-9)
        # ============================================
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        # ============================================
        # Domain Classifier: Source vs Target (MNIST vs MNIST-M)
        # ============================================
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=1.0):
        """
        Forward pass through DANN model.

        Args:
            input_data: Input images (batch, 3, 28, 28)
            alpha: Gradient reversal strength (0=no reversal, 1=full reversal)

        Returns:
            class_output: Digit predictions (batch, 10)
            domain_output: Domain predictions (batch, 2)
        """
        # Extract features
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)

        # Apply gradient reversal for domain classification
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        # Class prediction (digit classification)
        class_output = self.class_classifier(feature)

        # Domain prediction (source vs target)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


# ============================================
# MNIST-M Dataset (Custom dataset for MNIST-M)
# ============================================
class MNISTM(torch.utils.data.Dataset):
    """
    MNIST-M dataset loader.

    MNIST-M is MNIST digits blended with random patches from BSDS500 dataset.
    Images are stored as PNG files with labels in text files.
    """

    def __init__(self, root, train=True, transform=None):
        """
        Initialize MNIST-M dataset.

        Args:
            root: Root directory containing MNIST-M data
            train: Whether to load training or test set
            transform: Optional transform to apply
        """
        self.root = Path(root)
        self.train = train
        self.transform = transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found. Please ensure MNIST-M data is available.')

        # Load labels
        label_file = self.root / ('mnist_m_train_labels.txt' if train else 'mnist_m_test_labels.txt')
        self.image_folder = self.root / ('mnist_m_train' if train else 'mnist_m_test')

        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                self.samples.append((img_name, int(label)))

    def _check_exists(self):
        """Check if dataset files exist."""
        train_file = self.root / 'mnist_m_train_labels.txt'
        test_file = self.root / 'mnist_m_test_labels.txt'
        train_folder = self.root / 'mnist_m_train'
        test_folder = self.root / 'mnist_m_test'
        return (train_file.exists() and test_file.exists() and
                train_folder.exists() and test_folder.exists())

    def __getitem__(self, index):
        """Get a single item."""
        img_name, target = self.samples[index]
        img_path = self.image_folder / img_name

        # Load image
        img = datasets.folder.pil_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        """Return dataset size."""
        return len(self.samples)


# ============================================
# Data Loading Functions
# ============================================
def get_mnist_loaders(batch_size=64, image_size=28):
    """
    Get MNIST data loaders.

    Args:
        batch_size: Batch size for data loading
        image_size: Image size (should be 28 for MNIST)

    Returns:
        train_loader, test_loader: DataLoaders for MNIST
    """
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(3),  # Convert to 3 channels for consistency with MNIST-M
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Load MNIST dataset (use local data if available, download otherwise)
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


def get_mnistm_loaders(batch_size=64, image_size=28):
    """
    Get MNIST-M data loaders.

    Args:
        batch_size: Batch size for data loading
        image_size: Image size (should be 28 for MNIST-M)

    Returns:
        train_loader, test_loader: DataLoaders for MNIST-M
    """
    # Data transforms (same as MNIST)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Load MNIST-M dataset (using local data in domain_adapt/data)
    train_dataset = MNISTM(
        root='./data/mnist_m',
        train=True,
        transform=transform
    )

    test_dataset = MNISTM(
        root='./data/mnist_m',
        train=False,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


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

        if mode == 'train':
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
def train_dann(source_name='mnist', target_name='mnist_m', n_epoch=100,
               batch_size=64, lr=1e-3, save_interval=10, zeta=1.0):
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
        source_train_loader, source_test_loader = get_mnist_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported source domain: {source_name}")

    if target_name == 'mnist_m':
        target_train_loader, target_test_loader = get_mnistm_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported target domain: {target_name}")

    print(f"Source train size: {len(source_train_loader.dataset)}")
    print(f"Target train size: {len(target_train_loader.dataset)}")

    # Create model
    model = CNNModel()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_class = nn.NLLLoss()
    loss_domain = nn.NLLLoss()

    # Move to device
    model = model.to(device)
    loss_class = loss_class.to(device)
    loss_domain = loss_domain.to(device)

    # Create model directory
    model_dir = Path('./models')
    model_dir.mkdir(exist_ok=True)

    # Create training tracker
    tracker = create_tracker(save_dir="./models_mnist")

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
            optimizer, loss_class, loss_domain, device, epoch, n_epoch, 
            mode = 'eval'
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

    print(f"\nAll results saved to {tracker.save_dir}/")


if __name__ == "__main__":
    train_dann(n_epoch=5, batch_size=64)  # Quick test run
