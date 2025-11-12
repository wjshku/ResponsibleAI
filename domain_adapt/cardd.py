#!/usr/bin/env python3
"""
Domain Adversarial Neural Network (DANN) for SD2 → Kontext adaptation.

This implements DANN for car damage detection domain adaptation:
- Source domain: SD2 (Stable Diffusion 2 generated images)
- Target domain: Kontext (Kontext generated images)

The model learns to:
1. Classify car damage (damaged/not damaged) on source domain
2. Become domain-invariant using gradient reversal
3. Maintain classification accuracy on target domain (Kontext)
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import os
import glob
import random
from typing import Optional
import argparse

from dann.train_dann import *
from dann.loader_dann import AbstractDANNDataset, AbstractDANNLoader, create_loader_function
from dann.model_dann import AbstractDANN

# Domain name mapping for folder paths
DOMAIN_FOLDER_MAP = {
    'sd2': 'SD2',
    'kontext': 'Kontext',
    'qwen': 'Qwen'
}


# ============================================
# CARDD Dataset (Custom dataset for CARDD)
# ============================================
class CARDDDataset(AbstractDANNDataset):
    """
    Dataset class for loading CARDD images as binary classification samples.

    Each image (both original and processed) is treated as a separate sample:
    - Original images have label 0 (no damage)
    - Processed images have label 1 (damaged)

    Enhanced with robust path handling, sampling, and memory loading capabilities.
    """

    def __init__(self, root, domain='sd2', train=True, transform=None, sample_size=None, random_seed=42, load_to_memory=True):
        """
        Initialize the CARDD dataset for binary classification.

        Args:
            root: Root directory containing GenAI_Results
            domain: Domain name ('sd2' or 'kontext')
            train: Whether to load training or test set
            transform: Optional transform to apply
            sample_size: If provided, load only this many random samples
            random_seed: Random seed for reproducible sampling
            load_to_memory: If True, load all images into memory at initialization
        """
        super().__init__(root, train, transform)
        self.domain = domain.lower()
        self.train = train
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.load_to_memory = load_to_memory
        self._images_in_memory = None  # Will store loaded images if load_to_memory=True

        if self.domain not in ['sd2', 'kontext']:
            raise ValueError(f"Domain must be 'sd2' or 'kontext', got '{domain}'")

        # Set split name
        split = 'CarDD-TR' if train else 'CarDD-VAL'

        # Construct data and metadata directories
        folder_name = DOMAIN_FOLDER_MAP.get(self.domain, self.domain.upper())
        self.data_dir = os.path.join(root, 'GenAI_Results', folder_name, split)
        self.metadata_dir = os.path.join(self.data_dir, 'metadata')

        if not self._check_exists():
            raise RuntimeError(f'Dataset not found. Please ensure CARDD {domain} {split} data is available at {self.data_dir} with metadata in {self.metadata_dir}')

        # Load samples from JSON files - each image becomes a separate sample
        self.samples = []
        json_pattern = os.path.join(self.metadata_dir, 'processing_*.json')
        json_files = glob.glob(json_pattern)

        print(f"Loading image paths from {len(json_files)} JSON files...")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)

                # Only process successful entries
                if not metadata.get('success', False):
                    continue

                # Get processed image path
                processed_path = metadata.get('processed_image_path', '')
                if processed_path:
                    # Extract just the filename from the path
                    processed_filename = os.path.basename(processed_path)
                    # Build full path using data_dir
                    full_processed_path = os.path.join(self.data_dir, processed_filename)

                    # Get original image path from metadata
                    original_path = metadata.get('original_image_path', '')

                    # Add original image sample (label 0)
                    if original_path:
                        self.samples.append((original_path, 0))

                    # Add processed image sample (label 1) if it exists
                    if os.path.exists(full_processed_path):
                        self.samples.append((full_processed_path, 1))

            except (json.JSONDecodeError, KeyError) as e:
                # Skip invalid JSON files
                continue

        # Apply sampling if requested
        if sample_size is not None and len(self.samples) > sample_size:
            random.seed(random_seed)
            self.samples = random.sample(self.samples, sample_size)

        print(f"Successfully loaded {len(self.samples)} CARDD samples (original: label 0, processed: label 1)")

        # Load all images to memory if requested
        if self.load_to_memory:
            print("Loading all images to memory (this may take a while)...")
            self._load_dataset_to_memory()
            print("All images loaded to memory successfully.")

    def _check_exists(self):
        """Check if dataset files exist."""
        return (os.path.exists(self.data_dir) and
                os.path.exists(self.metadata_dir) and
                len(glob.glob(os.path.join(self.metadata_dir, 'processing_*.json'))) > 0)

    def _remap_path_for_ec2(self, image_path: str) -> str:
        """
        Remap image paths for EC2 environment.

        The JSON metadata contains local paths, but on EC2 the data is synced
        to different locations. This method detects EC2 and remaps paths accordingly.
        """
        # Check if we're running on EC2 (common EC2 indicators)
        is_ec2 = (
            os.path.exists('/home/ubuntu') or  # Ubuntu EC2 default
            'ec2' in os.uname().nodename.lower() or  # hostname contains 'ec2'
            os.path.exists('/opt/aws')  # AWS tools installed
        )

        if not is_ec2:
            return image_path  # Use original path for local development

        # On EC2, remap the paths
        # Original local path: /Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image/filename.jpg
        # Should become: /home/ubuntu/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image/filename.jpg

        # Also handle GenAI processed images that might reference original CarDD paths
        if '/Users/wjs/Local Storage/CarDD_release/' in image_path:
            # Replace with EC2 path
            ec2_path = image_path.replace(
                '/Users/wjs/Local Storage/CarDD_release/',
                '/home/ubuntu/ResponsibleAI/CarDD_release/'
            )
            return ec2_path

        # For GenAI processed images, they should already be in the correct location
        # relative to the data_dir, but let's make sure
        if image_path.startswith('/'):
            # Absolute path - check if it exists, if not try relative to data_dir
            if not os.path.exists(image_path):
                # Try relative to data_dir
                rel_path = os.path.relpath(image_path, '/')
                candidate_path = os.path.join(self.data_dir, rel_path)
                if os.path.exists(candidate_path):
                    return candidate_path

        return image_path  # Return original if no remapping needed

    def _load_dataset_to_memory(self):
        """Load all images into memory with transform applied."""
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, desc: x  # Fallback if tqdm not available

        self._images_in_memory = []

        # Load all images
        for idx in tqdm(range(len(self.samples)), desc="Loading images to memory"):
            image_path, label = self.samples[idx]
            # Remap paths for EC2 environment
            image_path = self._remap_path_for_ec2(image_path)

            # Load image
            try:
                # Expand ~ in path to actual home directory
                expanded_path = os.path.expanduser(image_path)
                img = datasets.folder.pil_loader(expanded_path)
            except Exception as e:
                # If image loading fails, return a black PIL image
                img = Image.new('RGB', (224, 224), color='black')
                print(f"Error loading image {image_path}: {e}")

            # Apply transform if provided
            if self.transform is not None:
                img = self.transform(img)

            self._images_in_memory.append(img)

    def __getitem__(self, index):
        """Get a single data item as (image, label)."""
        image_path, label = self.samples[index]

        # Load image from memory or disk
        if self._images_in_memory is not None and index < len(self._images_in_memory):
            # Load from memory
            img = self._images_in_memory[index]
        else:
            # Load from disk
            # Remap paths for EC2 environment
            image_path = self._remap_path_for_ec2(image_path)

            # Load image using dataset PIL loader (compatible with torchvision transforms)
            try:
                # Expand ~ in path to actual home directory
                expanded_path = os.path.expanduser(image_path)
                img = datasets.folder.pil_loader(expanded_path)
            except Exception as e:
                # If image loading fails, return a black PIL image
                img = Image.new('RGB', (224, 224), color='black')
                print(f"Error loading image {image_path}: {e}")

            # Apply transform if provided
            if self.transform is not None:
                img = self.transform(img)

        return img, label

    def __len__(self):
        """Return dataset size."""
        return len(self.samples)


# ============================================
# DANN Model (CNN for CARDD)
# ============================================
class CARDDModel(AbstractDANN):
    """
    CNN Model for DANN with SD2/Kontext adaptation.
    
    Architecture adapted for car damage detection:
    - Feature Extractor: ResNet-like CNN backbone
    - Label Classifier: Binary classification (damaged/not damaged)
    - Domain Classifier: Binary classification (SD2/Kontext)
    """

    def __init__(self, num_classes=2, input_size=224):
        """
        Initialize CARDD model.
        
        Args:
            num_classes: Number of label classes (default: 2 for binary classification)
            input_size: Input image size (default: 224)
        """
        self.num_classes = num_classes
        self.input_size = input_size
        super(CARDDModel, self).__init__()

    def _build_feature_extractor(self) -> nn.Module:
        """
        Build the feature extraction CNN backbone with flattening.
        
        Uses a ResNet-like architecture adapted for car damage detection.
        
        Returns:
            Feature extractor network that outputs flattened features
        """
        feature = nn.Sequential()
        
        # First conv block
        feature.add_module('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3))
        feature.add_module('bn1', nn.BatchNorm2d(64))
        feature.add_module('relu1', nn.ReLU(True))
        feature.add_module('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # Second conv block
        feature.add_module('conv2', nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        feature.add_module('bn2', nn.BatchNorm2d(128))
        feature.add_module('relu2', nn.ReLU(True))
        feature.add_module('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # Third conv block
        feature.add_module('conv3', nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        feature.add_module('bn3', nn.BatchNorm2d(256))
        feature.add_module('relu3', nn.ReLU(True))
        feature.add_module('pool3', nn.AdaptiveAvgPool2d((4, 4)))
        
        # Flatten
        feature.add_module('flatten', nn.Flatten())
        
        return feature

    def _build_label_classifier(self) -> nn.Module:
        """
        Build the label classification head (damage detection: damaged/not damaged).
        
        Returns:
            Label classifier network (output: 2 classes with LogSoftmax)
        """
        feature_dim = self._get_feature_dim()
        class_classifier = nn.Sequential()
        class_classifier.add_module('fc1', nn.Linear(feature_dim, 512))
        class_classifier.add_module('bn1', nn.BatchNorm1d(512))
        class_classifier.add_module('relu1', nn.ReLU(True))
        class_classifier.add_module('drop1', nn.Dropout(0.2))
        class_classifier.add_module('fc2', nn.Linear(512, 256))
        class_classifier.add_module('bn2', nn.BatchNorm1d(256))
        class_classifier.add_module('relu2', nn.ReLU(True))
        class_classifier.add_module('drop2', nn.Dropout(0.2))
        class_classifier.add_module('fc3', nn.Linear(256, self.num_classes))
        class_classifier.add_module('softmax', nn.LogSoftmax(dim=1))
        return class_classifier

    def _build_domain_classifier(self) -> nn.Module:
        """
        Build the domain classification head (SD2 vs Kontext).

        Returns:
            Domain classifier network (output: 2 classes with LogSoftmax)
        """
        feature_dim = self._get_feature_dim()
        domain_classifier = nn.Sequential()
        domain_classifier.add_module('d_fc1', nn.Linear(feature_dim, 512))
        domain_classifier.add_module('d_bn1', nn.BatchNorm1d(512))
        domain_classifier.add_module('d_relu1', nn.ReLU(True))
        domain_classifier.add_module('d_drop1', nn.Dropout(0.2))
        domain_classifier.add_module('d_fc2', nn.Linear(512, 256))
        domain_classifier.add_module('d_bn2', nn.BatchNorm1d(256))
        domain_classifier.add_module('d_relu2', nn.ReLU(True))
        domain_classifier.add_module('d_drop2', nn.Dropout(0.2))
        domain_classifier.add_module('d_fc3', nn.Linear(256, 2))
        domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        return domain_classifier

    def _get_feature_dim(self) -> int:
        """
        Get the flattened feature dimension.
        
        Feature extractor output: 256 channels * 4 * 4 = 4096
        
        Returns:
            Flattened feature dimension (4096)
        """
        return 256 * 4 * 4


# ============================================
# Data Loading Classes (using Abstract Base Classes)
# ============================================
class SD2Loader(AbstractDANNLoader):
    """
    SD2 data loader class using AbstractDANNLoader.
    """

    def __init__(self, data_root: str = '../cardd_data'):
        """
        Initialize SD2 loader.

        Args:
            data_root: Root directory for CARDD data
        """
        self.data_root = Path(data_root)

    def get_loaders(self, batch_size=32, image_size=224, sample_size=None, random_seed=42, load_to_memory=True, **kwargs):
        """
        Get SD2 data loaders.

        Args:
            batch_size: Batch size for data loading
            image_size: Image size (default: 224)
            sample_size: If provided, load only this many random samples per dataset
            random_seed: Random seed for reproducible sampling
            load_to_memory: If True, load all images into memory at initialization
            **kwargs: Additional arguments

        Returns:
            train_loader, test_loader: DataLoaders for SD2
        """
        # Create transforms
        transform = self.create_default_transform(
            image_size=image_size,
            grayscale_to_rgb=False,  # Images are already RGB
            normalize_mean=(0.485, 0.456, 0.406),  # ImageNet normalization
            normalize_std=(0.229, 0.224, 0.225)
        )

        # Load SD2 dataset
        train_dataset = CARDDDataset(
            root=str(self.data_root),
            domain='sd2',
            train=True,
            transform=transform,
            sample_size=sample_size,
            random_seed=random_seed,
            load_to_memory=load_to_memory
        )

        test_dataset = CARDDDataset(
            root=str(self.data_root),
            domain='sd2',
            train=False,
            transform=transform,
            sample_size=500 if sample_size is not None else sample_size,
            random_seed=random_seed,
            load_to_memory=load_to_memory
        )

        # Create data loaders
        train_loader = self.create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = self.create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, test_loader


class KontextLoader(AbstractDANNLoader):
    """
    Kontext data loader class using AbstractDANNLoader.
    """

    def __init__(self, data_root: str = '../cardd_data'):
        """
        Initialize Kontext loader.

        Args:
            data_root: Root directory for CARDD data
        """
        self.data_root = Path(data_root)

    def get_loaders(self, batch_size=32, image_size=224, sample_size=None, random_seed=42, load_to_memory=True, **kwargs):
        """
        Get Kontext data loaders.

        Args:
            batch_size: Batch size for data loading
            image_size: Image size (default: 224)
            sample_size: If provided, load only this many random samples per dataset
            random_seed: Random seed for reproducible sampling
            load_to_memory: If True, load all images into memory at initialization
            **kwargs: Additional arguments

        Returns:
            train_loader, test_loader: DataLoaders for Kontext
        """
        # Create transforms
        transform = self.create_default_transform(
            image_size=image_size,
            grayscale_to_rgb=False,  # Images are already RGB
            normalize_mean=(0.485, 0.456, 0.406),  # ImageNet normalization
            normalize_std=(0.229, 0.224, 0.225)
        )

        # Load Kontext dataset
        train_dataset = CARDDDataset(
            root=str(self.data_root),
            domain='kontext',
            train=True,
            transform=transform,
            sample_size=sample_size,
            random_seed=random_seed,
            load_to_memory=load_to_memory
        )

        test_dataset = CARDDDataset(
            root=str(self.data_root),
            domain='kontext',
            train=False,
            transform=transform,
            sample_size=500 if sample_size is not None else sample_size,
            random_seed=random_seed,
            load_to_memory=load_to_memory
        )

        # Create data loaders
        train_loader = self.create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = self.create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN for CARDD SD2 → Kontext adaptation')
    parser.add_argument('--sample_size', type=int, default=2000, help='Number of samples to use for training (test uses 500 if sampling enabled)')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size (square)')
    parser.add_argument('--n_epoch', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=5.0, help='Domain adaptation parameter gamma')
    parser.add_argument('--zeta', type=float, default=1.0, help='Domain adaptation parameter zeta')
    parser.add_argument('--save_dir', type=str, default="./models_cardd", help='Directory to save trained models')

    args = parser.parse_args()

    # ============================================
    source_name, target_name = 'sd2', 'kontext'
    def get_src_loaders(batch_size=64, **kwargs):
        loader = SD2Loader()
        return loader.get_loaders(batch_size=batch_size, image_size=args.image_size, load_to_memory=True, sample_size=args.sample_size, random_seed=42, **kwargs)

    def get_tgt_loaders(batch_size=64, **kwargs):
        loader = KontextLoader()
        return loader.get_loaders(batch_size=batch_size, image_size=args.image_size, load_to_memory=True, sample_size=args.sample_size, random_seed=42, **kwargs)

    train_dann(
        n_epoch=args.n_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        zeta=args.zeta,
        save_dir=args.save_dir,
        source_name=source_name,
        get_src_loaders=get_src_loaders,
        target_name=target_name,
        get_tgt_loaders=get_tgt_loaders,
        input_model=CARDDModel
    )

