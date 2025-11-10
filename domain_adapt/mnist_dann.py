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
from pathlib import Path


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
def train_dann_epoch(model, source_loader, target_loader, optimizer,
                    loss_class, loss_domain, device, epoch, n_epoch):
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
        Average losses for the epoch
    """
    model.train()

    # Calculate lambda (adaptation strength) using schedule from DANN paper
    len_dataloader = min(len(source_loader), len(target_loader))
    total_batches = len_dataloader * n_epoch

    # Create iterators
    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_loader)

    total_class_loss = 0.0
    total_domain_loss = 0.0
    n_batches = 0

    for batch_idx in range(len_dataloader):
        # Calculate progress p and lambda
        p = float(batch_idx + epoch * len_dataloader) / total_batches
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

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

        # Forward pass
        model.zero_grad()
        class_output, domain_output = model(s_img, alpha)

        # Classification loss (digit classification)
        err_s_label = loss_class(class_output, s_label)

        # Domain loss (source domain = 0)
        domain_label = torch.zeros(batch_size).long().to(device)
        err_s_domain = loss_domain(domain_output, domain_label)

        # ============================================
        # Train on target data (MNIST-M)
        # ============================================
        try:
            target_data = next(data_target_iter)
        except StopIteration:
            data_target_iter = iter(target_loader)
            target_data = next(data_target_iter)

        t_img, _ = target_data
        batch_size = len(t_img)

        # Move to device
        t_img = t_img.to(device)

        # Forward pass (only domain classification for target)
        _, domain_output = model(t_img, alpha)

        # Domain loss (target domain = 1)
        domain_label = torch.ones(batch_size).long().to(device)
        err_t_domain = loss_domain(domain_output, domain_label)

        # Total loss
        err = err_s_label + err_s_domain + err_t_domain
        err.backward()
        optimizer.step()

        # Track losses
        total_class_loss += err_s_label.item()
        total_domain_loss += (err_s_domain.item() + err_t_domain.item())
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}/{len_dataloader}, '
                  f'Class Loss: {err_s_label.item():.4f}, Domain Loss: {(err_s_domain + err_t_domain).item():.4f}, '
                  f'Lambda: {alpha:.4f}')

    # Return average losses
    avg_class_loss = total_class_loss / n_batches
    avg_domain_loss = total_domain_loss / n_batches

    return avg_class_loss, avg_domain_loss


def test_model(model, test_loader, device, domain_name="test"):
    """
    Test model on a dataset.

    Args:
        model: CNNModel instance
        test_loader: DataLoader for test data
        device: Device (CPU/CUDA)
        domain_name: Name for logging

    Returns:
        Accuracy on the test set
    """
    model.eval()

    n_total = 0
    n_correct = 0

    with torch.no_grad():
        for data in test_loader:
            img, label = data
            batch_size = len(label)

            # Move to device
            img = img.to(device)
            label = label.to(device)

            # Convert grayscale to RGB if needed
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)

            # Forward pass (alpha=0 for inference)
            class_output, _ = model(img, alpha=0)

            # Get predictions
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()
            n_total += batch_size

    accuracy = n_correct * 1.0 / n_total
    print(f'Accuracy on {domain_name}: {accuracy:.4f}')

    return accuracy


# ============================================
# Main Training Function
# ============================================
def train_dann(source_name='mnist', target_name='mnist_m', n_epoch=100,
               batch_size=64, lr=1e-3, save_interval=10):
    """
    Train DANN model for MNIST → MNIST-M adaptation.

    Args:
        source_name: Source domain name ('mnist')
        target_name: Target domain name ('mnist_m')
        n_epoch: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        save_interval: Save model every N epochs
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

    # Training loop
    print("\nStarting training...")
    best_source_acc = 0.0
    best_target_acc = 0.0

    for epoch in range(n_epoch):
        print(f"\nEpoch {epoch+1}/{n_epoch}")
        print("-" * 50)

        # Train for one epoch
        class_loss, domain_loss = train_dann_epoch(
            model, source_train_loader, target_train_loader,
            optimizer, loss_class, loss_domain, device, epoch, n_epoch
        )

        # Test on both domains
        source_acc = test_model(model, source_test_loader, device, f"{source_name} test")
        target_acc = test_model(model, target_test_loader, device, f"{target_name} test")

        # Save best models
        if source_acc > best_source_acc:
            best_source_acc = source_acc
            torch.save(model.state_dict(), model_dir / 'best_source_model.pth')

        if target_acc > best_target_acc:
            best_target_acc = target_acc
            torch.save(model.state_dict(), model_dir / 'best_target_model.pth')

        # Save model periodically
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), model_dir / f'model_epoch_{epoch+1}.pth')

        print(".4f"
              ".4f"
              ".4f")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print(".4f")
    print(".4f")
    print("=" * 70)


if __name__ == "__main__":
    train_dann(n_epoch=2, batch_size=32)  # Quick test run
