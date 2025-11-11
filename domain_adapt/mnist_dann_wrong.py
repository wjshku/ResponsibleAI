#!/usr/bin/env python3
"""
Test DANN workflow on MNIST vs MNIST-M dataset.

This script verifies that our DANN implementation works on the classic
domain adaptation benchmark from the original DANN paper.

MNIST-M: MNIST digits blended with random patches from BSDS500 dataset
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Set matplotlib backend to avoid display issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add current directory to path


def setup_logging(save_dir, experiment_name):
    """Setup logging to both console and file."""
    # Setup logging - save log file directly in experiment directory
    log_filename = os.path.join(save_dir, f'{experiment_name}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Log experiment start
    logging.info(f"Starting experiment: {experiment_name}")
    logging.info(f"Results will be saved to: {save_dir}")

    return logging.getLogger()


# Add current directory to path
sys.path.append(str(Path(__file__).parent))
# We'll define our own components instead of importing from model_dann

# Import torch for loading MNIST-M data
import torch


# ============================================
# Gradient Reversal Layer Components
# ============================================
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) from the DANN paper.

    Forward pass: Identity function (output = input)
    Backward pass: Multiply gradient by -lambda (reversed gradient)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """Forward pass: Identity transformation."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: Reverse and scale gradient."""
        lambda_ = ctx.lambda_
        grad_input = -lambda_ * grad_output
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer as a PyTorch module."""

    def __init__(self):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = 1.0

    def forward(self, x):
        """Apply gradient reversal with current lambda value."""
        return GradientReversalFunction.apply(x, self.lambda_)


# ============================================
# MNIST-M Dataset Creation
# ============================================
class MNISTMDataset(Dataset):
    """
    MNIST-M dataset: MNIST digits with BSDS500 backgrounds.

    Your format: Images in folders + labels in .txt files
    """

    def __init__(self, root='./data', train=True, transform=None, max_samples=None):
        """
        Args:
            root: Root directory for data
            train: Whether to load training set
            transform: Optional transform
            max_samples: Maximum number of samples to load (for quick testing)
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.max_samples = max_samples

        # Load MNIST-M data from your format
        self.data, self.labels = self._load_mnist_m()

    def _load_mnist_m(self):
        """Load MNIST-M dataset from your folder + txt format."""
        import os
        from PIL import Image

        # Determine paths
        data_split = 'train' if self.train else 'test'
        img_dir = os.path.join(self.root, 'mnist_m', f'mnist_m_{data_split}')
        label_file = os.path.join(self.root, 'mnist_m', f'mnist_m_{data_split}_labels.txt')

        if not os.path.exists(img_dir) or not os.path.exists(label_file):
            raise FileNotFoundError(f"MNIST-M {data_split} data not found at {img_dir} or {label_file}")

        # Load labels from txt file
        labels_dict = {}
        with open(label_file, 'r') as f:
            for line in f:
                filename, label = line.strip().split()
                labels_dict[filename] = int(label)

        # Apply max_samples by randomly sampling files (much faster for quick testing)
        if self.max_samples is not None and len(labels_dict) > self.max_samples:
            import random
            sampled_items = random.sample(list(labels_dict.items()), self.max_samples)
            labels_dict = dict(sampled_items)

        # Load images and create tensors
        data_list = []
        labels_list = []

        print(f"Loading MNIST-M {data_split} images ({len(labels_dict)} samples)...")
        for filename, label in labels_dict.items():
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                # Load image and resize to 28x28 to match MNIST dimensions
                img = Image.open(img_path).convert('RGB')
                img = img.resize((28, 28), Image.Resampling.LANCZOS)

                # Convert to tensor
                img_tensor = transforms.ToTensor()(img)

                data_list.append(img_tensor)
                labels_list.append(label)
            else:
                print(f"Warning: {img_path} not found")

        # Stack into tensors
        data = torch.stack(data_list)
        labels = torch.tensor(labels_list)

        print(f"✅ Loaded MNIST-M {data_split}: {len(data)} samples")
        return data, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]

        # Note: images are already tensors, so we handle transforms manually
        if self.transform:
            for t in self.transform.transforms:
                if isinstance(t, transforms.ToTensor):
                    # Skip ToTensor since images are already tensors
                    continue
                elif isinstance(t, transforms.Normalize):
                    # Apply normalization directly to tensor
                    img = t(img)
                else:
                    # Apply other transforms normally (like Resize)
                    img = t(img)

        return img, label


# ============================================
# DANN Model for MNIST/MNIST-M
# ============================================
class MNISTDANN(nn.Module):
    """DANN for MNIST/MNIST-M domain adaptation (digits 0-9).

    Architecture matches the DANN paper:
    - Feature extractor: 2 conv layers (64 and 50 maps) with batch norm and dropout
    - Label predictor: 2 FC layers (100 units each) with batch norm and dropout
    - Domain classifier: 1 FC layer (100 units) with batch norm
    """

    def __init__(self, num_classes=10):
        super(MNISTDANN, self).__init__()

        # Feature extractor (matches DANN paper architecture)
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature_extractor.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature_extractor.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature_extractor.add_module('f_relu1', nn.ReLU(True))
        self.feature_extractor.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature_extractor.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature_extractor.add_module('f_drop1', nn.Dropout2d())
        self.feature_extractor.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature_extractor.add_module('f_relu2', nn.ReLU(True))
        self.feature_extractor.add_module('f_flatten', nn.Flatten())


        # Calculate feature dimension: 50 * 4 * 4 = 800
        self.feature_dim = 50 * 4 * 4

        # Label predictor (digit classification)
        self.label_predictor = nn.Sequential()
        self.label_predictor.add_module('c_fc1', nn.Linear(self.feature_dim, 100))
        self.label_predictor.add_module('c_bn1', nn.BatchNorm1d(100))
        self.label_predictor.add_module('c_relu1', nn.ReLU(True))
        self.label_predictor.add_module('c_drop1', nn.Dropout())
        self.label_predictor.add_module('c_fc2', nn.Linear(100, 100))
        self.label_predictor.add_module('c_bn2', nn.BatchNorm1d(100))
        self.label_predictor.add_module('c_relu2', nn.ReLU(True))
        self.label_predictor.add_module('c_fc3', nn.Linear(100, num_classes))
        # Softmax will be applied in loss function (CrossEntropyLoss includes it)

        # Domain classifier (MNIST vs MNIST-M)
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.feature_dim, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        # Softmax will be applied in loss function (CrossEntropyLoss includes it)

        # Gradient reversal layer
        self.grl = GradientReversalLayer()

    def forward(self, x, lambda_=1.0):
        """
        Forward pass.

        Args:
            x: Input images (3-channel RGB)
            lambda_: Adaptation strength

        Returns:
            label_output: Digit classification logits
            domain_output: Domain classification logits
        """

        # Extract features
        features = self.feature_extractor(x)

        # Label prediction (digit classification)
        label_output = self.label_predictor(features)

        # Domain prediction (with gradient reversal)
        self.grl.lambda_ = lambda_
        reversed_features = self.grl(features)
        domain_output = self.domain_classifier(reversed_features)

        return label_output, domain_output


# ============================================
# Training Functions
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


def train_epoch(model, source_loader, target_loader, optimizer,
                criterion_label, criterion_domain, device, epoch, total_epochs, config):
    """Train one epoch of DANN."""

    model.train()
    total_label_loss = 0.0
    total_domain_loss = 0.0
    total_label_correct = 0
    total_domain_correct = 0
    total_samples = 0

    # Calculate lambda for this epoch (constant throughout epoch)
    lambda_ = compute_lambda_schedule(epoch, total_epochs)

    # Create iterators
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    # Train until shorter dataset is exhausted
    num_batches = min(len(source_loader), len(target_loader))

    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{total_epochs}")
    pbar.set_postfix({'λ': f'{lambda_:.3f}'})

    for batch_idx in pbar:

        # Get source batch (labeled)
        try:
            batch = next(source_iter)
            if len(batch) == 3:  # (images, labels, domain)
                source_images, source_labels, _ = batch
            else:  # (images, labels)
                source_images, source_labels = batch
        except StopIteration:
            source_iter = iter(source_loader)
            batch = next(source_iter)
            if len(batch) == 3:
                source_images, source_labels, _ = batch
            else:
                source_images, source_labels = batch

        source_images = source_images.to(device)
        source_labels = source_labels.to(device)

        # Get target batch (unlabeled for domain adaptation)
        try:
            batch = next(target_iter)
            if len(batch) == 3:  # (images, labels, domain)
                target_images, _, _ = batch
            else:  # (images, labels)
                target_images, _ = batch
        except StopIteration:
            target_iter = iter(target_loader)
            batch = next(target_iter)
            if len(batch) == 3:
                target_images, _, _ = batch
            else:
                target_images, _ = batch

        target_images = target_images.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass for source domain
        source_label_pred, source_domain_pred = model(source_images, lambda_)

        # Forward pass for target domain
        target_label_pred, target_domain_pred = model(target_images, lambda_)

        # Label loss (only on source domain)
        label_loss = criterion_label(source_label_pred, source_labels)

        # Domain loss (on both domains)
        # Binary classification: 0 for source (MNIST), 1 for target (MNIST-M)
        source_domain_labels = torch.zeros(source_images.size(0), dtype=torch.long, device=device)
        target_domain_labels = torch.ones(target_images.size(0), dtype=torch.long, device=device)

        domain_loss = criterion_domain(source_domain_pred, source_domain_labels) + \
                     criterion_domain(target_domain_pred, target_domain_labels)

        # Total loss
        total_loss = label_loss + domain_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistics
        total_label_loss += label_loss.item()
        total_domain_loss += domain_loss.item()

        # Label accuracy (source only)
        _, predicted_labels = torch.max(source_label_pred, 1)
        total_label_correct += (predicted_labels == source_labels).sum().item()

        # Domain accuracy (both domains)
        # Use argmax for multi-class classification
        _, predicted_source_domains = torch.max(source_domain_pred, 1)
        _, predicted_target_domains = torch.max(target_domain_pred, 1)

        total_domain_correct += (predicted_source_domains == source_domain_labels).sum().item()
        total_domain_correct += (predicted_target_domains == target_domain_labels).sum().item()

        total_samples += source_images.size(0) + target_images.size(0)

    # Close progress bar
    pbar.close()

    # Return averages
    avg_label_loss = total_label_loss / num_batches
    avg_domain_loss = total_domain_loss / num_batches
    label_accuracy = total_label_correct / (total_samples // 2)  # Only source samples
    domain_accuracy = total_domain_correct / total_samples

    return {
        'label_loss': avg_label_loss,
        'domain_loss': avg_domain_loss,
        'label_accuracy': label_accuracy,
        'domain_accuracy': domain_accuracy,
        'lambda': lambda_
    }


def evaluate(model, dataloader, device, domain_name="Unknown", criterion=None):
    """Evaluate model on a dataset."""

    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  # Training format: (images, labels, domain)
                images, labels, _ = batch
            else:  # Test format: (images, labels)
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            # Only use label prediction (no domain adaptation during eval)
            label_outputs, _ = model(images, lambda_=0.0)

            _, predicted = torch.max(label_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(label_outputs, labels)
                total_loss += loss.item() * labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / total if criterion is not None else None

    if criterion is not None:
        print(f"{domain_name} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy, avg_loss
    else:
        print(f"{domain_name} Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy


def plot_training_losses(history, save_dir, logger=None):
    """Plot and save training loss curves for DANN adversarial training."""
    import matplotlib.pyplot as plt

    if logger is None:
        logger = logging.getLogger()

    # Extract data from history
    epochs = [h['epoch'] for h in history]
    label_losses = [h['label_loss'] for h in history]
    domain_losses = [h['domain_loss'] for h in history]
    total_losses = [h['label_loss'] + h['domain_loss'] for h in history]
    lambdas = [h['lambda'] for h in history]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DANN Training: Class Classifier vs Domain Classifier Losses', fontsize=16, fontweight='bold')

    # Plot 1: Main losses (Class vs Domain Classifier)
    ax1.plot(epochs, label_losses, 'b-', linewidth=2, label='Class Classifier Loss', marker='o', markersize=4)
    ax1.plot(epochs, domain_losses, 'r-', linewidth=2, label='Domain Classifier Loss', marker='s', markersize=4)
    ax1.set_title('Adversarial Losses (Generator vs Discriminator)', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total loss
    ax2.plot(epochs, total_losses, 'g-', linewidth=2, label='Total Loss', marker='^', markersize=4)
    ax2.set_title('Total Training Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Individual losses with different scales
    ax3.plot(epochs, label_losses, 'b-', linewidth=2, label='Class Loss', marker='o', markersize=4)
    ax3.set_title('Class Classifier Loss', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.plot(epochs, domain_losses, 'r-', linewidth=2, label='Domain Loss', marker='s', markersize=4)
    ax4.set_title('Domain Classifier Loss', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, 'dann_training_losses.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    # Close the figure to free memory
    plt.close(fig)

    # Create a second plot focusing on the adversarial dynamics (lambda schedule)
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax2 = ax.twinx()

    # Plot losses
    line1 = ax.plot(epochs, label_losses, 'b-', linewidth=2, label='Class Classifier (Generator)', marker='o', markersize=4)
    line2 = ax.plot(epochs, domain_losses, 'r-', linewidth=2, label='Domain Classifier (Discriminator)', marker='s', markersize=4)

    # Plot lambda schedule
    line3 = ax2.plot(epochs, lambdas, 'g--', linewidth=2, label='Lambda (GRL Strength)', marker='^', markersize=4)

    ax.set_title('DANN Adversarial Training Dynamics', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='black')
    ax2.set_ylabel('Lambda (Gradient Reversal Strength)', color='green')

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')

    ax.grid(True, alpha=0.3)

    # Save the adversarial dynamics plot
    plot_path2 = os.path.join(save_dir, 'dann_adversarial_dynamics.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')

    plt.close(fig2)


# ============================================
# Main Training Script
# ============================================
def train_baseline(source_loader, target_loader, config, num_epochs=5):
    """Train a baseline model without domain adaptation."""
    print("\n📊 Training Baseline Model (No Domain Adaptation)...")

    model = MNISTDANN(num_classes=10)
    model.to(config['device'])

    criterion = nn.CrossEntropyLoss()
    # Baseline uses fixed learning rate of 0.001 (following original DANN paper)
    baseline_lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=baseline_lr)

    # Train only on source domain
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for progress bar
        with tqdm(source_loader, desc=f"Baseline Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, labels in pbar:
                images = images.to(config['device'])
                labels = labels.to(config['device'])

                optimizer.zero_grad()
                outputs, _ = model(images, lambda_=0.0)  # No domain adaptation
                loss = criterion(outputs, labels)

                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss / (pbar.n + 1):.4f}",
                    'acc': f"{correct / total:.4f}"
                })

        print(f"Baseline Epoch {epoch+1}/{num_epochs}: Loss={total_loss/len(source_loader):.4f}, Acc={correct/total:.4f}")

    return model


def main():
    """Main training function."""

    print("🚀 Domain Adaptation: MNIST → MNIST-M")
    print("=" * 50)

    # Auto-detect best available device
    if torch.cuda.is_available():
        device = 'cuda'
        print("🎮 Using CUDA GPU")
    else:
        # Try MPS even if is_available() returns False (sometimes works in restricted environments)
        # try:
        #     test_device = torch.device('mps')
        #     # Test if MPS actually works by creating a small tensor
        #     test_tensor = torch.randn(1, device=test_device)
        #     device = 'mps'
        #     print("🍎 Using Apple Silicon MPS (tested and working)")
        # except Exception as e:
        #     device = 'cpu'
        #     print(f"💻 Using CPU (MPS not available: {str(e)[:50]}...)")
        device = 'cpu'
        
    # Check if this is a quick test run
    import sys
    quick_test = '--quick' in sys.argv

    # Create timestamped experiment directory in models_mnist
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'./models_mnist/model_dann_{timestamp}'

    config = {
        'batch_size': 128 if quick_test else 64,  # Larger batch for quick test
        'num_epochs': 2 if quick_test else 20,     # Quick test or full training
        'learning_rate': 1e-3,
        'image_size': 28,  # Image size for resizing
        'device': device,
        'save_dir': save_dir,
        'experiment_name': f'model_dann_{timestamp}',
        'timestamp': timestamp
    }

    # Create save directory first
    os.makedirs(config['save_dir'], exist_ok=True)

    # Setup logging
    logger = setup_logging(config['save_dir'], config['experiment_name'])

    logger.info(f"Using device: {config['device']}")
    logger.info(f"Configuration: {config}")

    # Data transforms for MNIST (grayscale)
    mnist_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.Grayscale(3),  # Convert to 3 channels for consistency with MNIST-M
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Data transforms for MNIST-M (RGB)
    mnist_m_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Create real MNIST and MNIST-M datasets
    logger.info("📦 Loading real MNIST and MNIST-M datasets...")

    # Set max_samples for quick testing
    max_samples = 200 if quick_test else None

    # Source domain: MNIST (grayscale)
    source_train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=mnist_transform
    )

    if max_samples is not None and len(source_train_dataset) > max_samples:
        from torch.utils.data import Subset
        indices = torch.randperm(len(source_train_dataset))[:max_samples]
        source_train_dataset = Subset(source_train_dataset, indices)

    source_test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=mnist_transform
    )

    if max_samples is not None and len(source_test_dataset) > max_samples:
        from torch.utils.data import Subset
        indices = torch.randperm(len(source_test_dataset))[:max_samples]
        source_test_dataset = Subset(source_test_dataset, indices)

    # Target domain: MNIST-M (color)
    target_train_dataset = MNISTMDataset(
        root='./data',
        train=True,
        transform=mnist_m_transform,
        max_samples=max_samples
    )

    target_test_dataset = MNISTMDataset(
        root='./data',
        train=False,
        transform=mnist_m_transform,
        max_samples=max_samples
    )

    # Create data loaders
    source_loader = DataLoader(source_train_dataset, batch_size=config['batch_size'], shuffle=True)
    target_loader = DataLoader(target_train_dataset, batch_size=config['batch_size'], shuffle=True)

    mnist_test_loader = DataLoader(source_test_dataset, batch_size=config['batch_size'], shuffle=False)
    mnist_m_test_loader = DataLoader(target_test_dataset, batch_size=config['batch_size'], shuffle=False)

    logger.info(f"MNIST train: {len(source_train_dataset)} samples")
    logger.info(f"MNIST-M train: {len(target_train_dataset)} samples")
    logger.info(f"MNIST test: {len(source_test_dataset)} samples")
    logger.info(f"MNIST-M test: {len(target_test_dataset)} samples")

    # Create model
    logger.info("🏗️ Creating DANN model...")
    model = MNISTDANN(num_classes=10)
    model.to(config['device'])

    # Loss functions
    criterion_label = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()  # Binary classification for domain (2 output units with softmax)

    # Optimizer (using Adam for better convergence)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Adam optimizer handles learning rate adaptively - no manual scheduling needed

    # Training history
    history = []

    # Train and evaluate baseline model (skip by default for faster DANN testing)
    skip_baseline = True  # Set to False to run baseline training
    if skip_baseline:
        logger.info("\n⏭️  Skipping baseline training (set skip_baseline=False to enable)")
        baseline_mnist_acc = 0.0
        baseline_mnist_m_acc = 0.0
    else:
        logger.info("\n🔍 Establishing Baseline...")
        baseline_model = train_baseline(source_loader, target_loader, config, num_epochs=5)
        baseline_mnist_acc, baseline_mnist_loss = evaluate(baseline_model, mnist_test_loader, config['device'], "MNIST (baseline)", criterion_label)
        baseline_mnist_m_acc, baseline_mnist_m_loss = evaluate(baseline_model, mnist_m_test_loader, config['device'], "MNIST-M (baseline)", criterion_label)

    logger.info(f"Baseline MNIST-M Accuracy: {baseline_mnist_m_acc:.4f}")
    # Evaluate before training DANN
    logger.info("\n📊 Pre-training DANN evaluation:")
    mnist_acc_before, mnist_loss_before = evaluate(model, mnist_test_loader, config['device'], "MNIST (before DANN)", criterion_label)
    mnist_m_acc_before, mnist_m_loss_before = evaluate(model, mnist_m_test_loader, config['device'], "MNIST-M (before DANN)", criterion_label)

    # Training loop
    logger.info("\n🎯 Starting DANN training...")
    logger.info(f"Training for {config['num_epochs']} epochs")
    for epoch in range(config['num_epochs']):
        logger.debug(f"Starting epoch {epoch+1}")

        # Train one epoch
        epoch_stats = train_epoch(
            model, source_loader, target_loader,
            optimizer, criterion_label, criterion_domain,
            config['device'], epoch, config['num_epochs'], config
        )

        # Adam optimizer handles learning rate adaptively

        # Evaluate on test sets
        mnist_acc, mnist_loss = evaluate(model, mnist_test_loader, config['device'], f"MNIST (epoch {epoch+1})", criterion_label)
        mnist_m_acc, mnist_m_loss = evaluate(model, mnist_m_test_loader, config['device'], f"MNIST-M (epoch {epoch+1})", criterion_label)

        # Store results
        epoch_result = {
            **epoch_stats,
            'epoch': epoch + 1,
            'mnist_test_acc': mnist_acc,
            'mnist_test_loss': mnist_loss,
            'mnist_m_test_acc': mnist_m_acc,
            'mnist_m_test_loss': mnist_m_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        history.append(epoch_result)

        current_lr = optimizer.param_groups[0]['lr']
        current_lambda = epoch_stats['lambda']
        logger.info(f"Epoch {epoch+1}: Label Loss={epoch_stats['label_loss']:.4f}, "
                    f"Label Acc={epoch_stats['label_accuracy']:.4f}, "
                    f"Domain Loss={epoch_stats['domain_loss']:.4f}, "
                    f"Domain Acc={epoch_stats['domain_accuracy']:.4f}, "
                    f"Lambda={current_lambda:.4f}, "
                    f"LR={current_lr:.6f}, "
                    f"MNIST-M Acc={mnist_m_acc:.4f}")

    # Final evaluation
    logger.info("\n🏁 Final evaluation:")
    mnist_acc_final, mnist_loss_final = evaluate(model, mnist_test_loader, config['device'], "MNIST (final)", criterion_label)
    mnist_m_acc_final, mnist_m_loss_final = evaluate(model, mnist_m_test_loader, config['device'], "MNIST-M (final)", criterion_label)

    # Save results
    results = {
        'config': config,
        'history': history,
        'final_results': {
            'mnist_accuracy': mnist_acc_final,
            'mnist_m_accuracy': mnist_m_acc_final,
            'improvement': mnist_m_acc_final - baseline_mnist_m_acc if not skip_baseline else None
        }
    }

    torch.save(results, os.path.join(config['save_dir'], 'mnist_dann_results.pth'))
    logger.info(f"\n💾 Results saved to {config['save_dir']}/mnist_dann_results.pth")

    # Save trained model
    model_path = os.path.join(config['save_dir'], 'dann_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_mnist_accuracy': mnist_acc_final,
        'final_mnist_m_accuracy': mnist_m_acc_final
    }, model_path)
    logger.info(f"💾 Model saved to {model_path}")

    # Plot and save loss curves
    plot_training_losses(history, config['save_dir'], logger)

    # Print training summary
    logger.info("\n📈 Training Summary:")
    logger.info("Epoch | Label Loss | Label Acc | Domain Loss | Domain Acc | MNIST Test | MNIST-M Test | Lambda | LR")
    logger.info("-" * 100)
    for h in history:
        logger.info(f"{h['epoch']:>3d}   | {h['label_loss']:>8.4f}   | {h['label_accuracy']:>8.4f}   | {h['domain_loss']:>10.4f}   | {h['domain_accuracy']:>9.4f}   | {h['mnist_test_acc']:>9.4f}   | {h['mnist_m_test_acc']:>10.4f}   | {h['lambda']:>5.3f}   | {h['learning_rate']:>6.6f}")

    # Save baseline results
    results['baseline_results'] = {
        'mnist_accuracy': baseline_mnist_acc,
        'mnist_m_accuracy': baseline_mnist_m_acc
    }

    logger.info("\n" + "="*60)
    logger.info("🎉 DOMAIN ADAPTATION RESULTS")
    logger.info("="*60)

    if not skip_baseline:
        logger.info("BASELINE (No Domain Adaptation):")
        logger.info(f"  MNIST:     {results['baseline_results']['mnist_accuracy']:.4f}")
        logger.info(f"  MNIST-M:   {results['baseline_results']['mnist_m_accuracy']:.4f}")
        logger.info("")

    logger.info("DANN (Domain Adversarial Neural Network):")
    logger.info(f"  MNIST:     {results['final_results']['mnist_accuracy']:.4f}")
    logger.info(f"  MNIST-M:   {results['final_results']['mnist_m_accuracy']:.4f}")

    if not skip_baseline and results['final_results']['improvement'] is not None:
        logger.info("IMPROVEMENT:")
        logger.info(f"  MNIST-M:   {results['final_results']['improvement']:.4f}")

    # Domain adaptation effectiveness (only if baseline was run)
    if not skip_baseline and results['final_results']['improvement'] is not None:
        dann_improvement = results['final_results']['improvement']
        if dann_improvement > 0.05:  # 5% improvement threshold
            logger.info("✅ EXCELLENT: Strong domain adaptation effect!")
        elif dann_improvement > 0.02:  # 2% improvement threshold
            logger.info("✅ GOOD: Domain adaptation is working!")
        elif dann_improvement > 0:
            logger.info("⚠️  MARGINAL: Slight improvement, may need tuning")
        else:
            logger.info("❌ FAILURE: No domain adaptation benefit")
    else:
        logger.info("ℹ️  Baseline training skipped - cannot evaluate improvement")

    # Overall performance
    final_acc = results['final_results']['mnist_m_accuracy']
    if final_acc > 0.85:
        logger.info("✅ HIGH ACCURACY: Model performs well on target domain")
    elif final_acc > 0.75:
        logger.info("✅ DECENT ACCURACY: Acceptable performance")
    else:
        logger.info("⚠️  LOW ACCURACY: May need more training or architecture changes")

    logger.info(f"\n💾 All results and model saved to: {config['save_dir']}")
    logger.info("Files saved:")
    logger.info(f"  - Results: mnist_dann_results.pth")
    logger.info(f"  - Model: dann_model.pth")
    logger.info(f"  - Loss plots: dann_training_losses.png, dann_adversarial_dynamics.png")
    logger.info(f"  - Logs: {config['experiment_name']}.log")


if __name__ == "__main__":
    main()