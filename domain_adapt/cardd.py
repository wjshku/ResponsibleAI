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
from dann.eval_dann import evaluate_dann_model
from dann.utils import select_model_interactively
from model_arch_lib.model_deep import CARDDModel

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

    def __init__(self, root, domain='sd2', train=True, split=None, transform=None, sample_size=None, random_seed=42, load_to_memory=True):
        """
        Initialize the CARDD dataset for binary classification.

        Args:
            root: Root directory containing GenAI_Results
            domain: Domain name ('sd2' or 'kontext' or 'qwen')
            train: Whether to load training or test set (deprecated, use split instead)
            split: Dataset split ('TR', 'VAL', or 'TE'). If None, uses train parameter
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

        if self.domain not in ['sd2', 'kontext', 'qwen']:
            raise ValueError(f"Domain must be 'sd2', 'kontext', or 'qwen', got '{domain}'")

        # Set split name - support TR, VAL, or TE
        if split is not None:
            split_upper = split.upper()
            if split_upper in ['TR', 'VAL', 'TE']:
                split = f'CarDD-{split_upper}'
            else:
                raise ValueError(f"Split must be 'TR', 'VAL', or 'TE', got '{split}'")
        else:
            # Backward compatibility: use train parameter
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

        print(f"Loading image paths (Split: {split} - Domain: {domain}) from {len(json_files)} JSON files...")

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
# Data Loading Classes (using Abstract Base Classes)
# ============================================
class CARDDLoader(AbstractDANNLoader):
    """
    Unified CARDD data loader class for all domains (SD2, Kontext, Qwen).
    """

    def __init__(self, domain: str, data_root: str = '../cardd_data'):
        """
        Initialize CARDD loader for a specific domain.

        Args:
            domain: Domain name ('sd2', 'kontext', or 'qwen')
            data_root: Root directory for CARDD data
        """
        if domain not in ['sd2', 'kontext', 'qwen']:
            raise ValueError(f"Domain must be 'sd2', 'kontext', or 'qwen', got '{domain}'")

        self.domain = domain
        self.data_root = Path(data_root)

    def get_loaders(self, batch_size=32, image_size=224, sample_size=None, random_seed=42, load_to_memory=True, split=None, **kwargs):
        """
        Get CARDD data loaders for the specified domain.

        Args:
            batch_size: Batch size for data loading
            image_size: Image size (default: 224)
            sample_size: If provided, load only this many random samples per dataset
            random_seed: Random seed for reproducible sampling
            load_to_memory: If True, load all images into memory at initialization
            split: Dataset split ('TR', 'VAL', or 'TE'). If None, uses default splits
            **kwargs: Additional arguments

        Returns:
            train_loader, test_loader: DataLoaders for the specified domain
        """
        # Create transforms
        transform = self.create_default_transform(
            image_size=image_size,
            grayscale_to_rgb=False,  # Images are already RGB
            normalize_mean=(0.485, 0.456, 0.406),  # ImageNet normalization
            normalize_std=(0.229, 0.224, 0.225)
        )

        # Load dataset based on split parameter
        if split is not None:
            train_dataset = None
            
            test_dataset = CARDDDataset(
                root=str(self.data_root),
                domain=self.domain,
                split=split,
                transform=transform,
                sample_size=500 if sample_size is not None else sample_size,
                random_seed=random_seed,
                load_to_memory=load_to_memory
            )

            # Create data loaders
            train_loader = None

            test_loader = self.create_dataloader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        else:
            # Default behavior: TR for train, VAL for test
            train_dataset = CARDDDataset(
                root=str(self.data_root),
                domain=self.domain,
                train=True,
                transform=transform,
                sample_size=sample_size,
                random_seed=random_seed,
                load_to_memory=load_to_memory
            )

            test_dataset = CARDDDataset(
                root=str(self.data_root),
                domain=self.domain,
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


# Backward compatibility: create aliases for the old class names
def SD2Loader(data_root: str = '../cardd_data'):
    """Create SD2 loader (backward compatibility)."""
    return CARDDLoader('sd2', data_root)

def KontextLoader(data_root: str = '../cardd_data'):
    """Create Kontext loader (backward compatibility)."""
    return CARDDLoader('kontext', data_root)

def QwenLoader(data_root: str = '../cardd_data'):
    """Create Qwen loader (backward compatibility)."""
    return CARDDLoader('qwen', data_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or Test DANN for CARDD domain adaptation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Mode: train or test (default: train)')
    parser.add_argument('--source', type=str, default='sd2', choices=['sd2', 'kontext', 'qwen'],
                       help='Source domain for domain adaptation (default: sd2)')
    parser.add_argument('--target', type=str, default='kontext', choices=['sd2', 'kontext', 'qwen'],
                       help='Target domain for domain adaptation (default: kontext)')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to use for training (None = use all data, test uses 500 if sampling enabled)')
    parser.add_argument('--target_size', type=int, default=224, help='Target image size (square)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=10.0, help='Domain adaptation parameter gamma')
    parser.add_argument('--zeta', type=float, default=1.0, help='Domain adaptation parameter zeta')
    parser.add_argument('--save_dir', type=str, default="./models_cardd", help='Directory to save/load trained models')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model for testing (optional when mode=test, will prompt for selection)')

    args = parser.parse_args()

    # Interactive model selection for testing
    if args.mode == 'test' and args.model_path is None:
        args.model_path = select_model_interactively(args.save_dir)

    # Validate that source and target domains are different
    if args.source == args.target:
        raise ValueError(f"Source domain ({args.source}) and target domain ({args.target}) must be different")

    # ============================================
    # Domain loader mapping
    DOMAIN_LOADER_MAP = {
        'sd2': SD2Loader,
        'kontext': KontextLoader,
        'qwen': QwenLoader
    }

    source_name, target_name = args.source, args.target

    def get_src_loaders(batch_size=64, **kwargs):
        loader_class = DOMAIN_LOADER_MAP[source_name]
        if loader_class is None:
            raise ValueError(f"Loader for domain '{source_name}' is not implemented yet")
        loader = loader_class()
        return loader.get_loaders(batch_size=batch_size, image_size=args.target_size, load_to_memory=True, sample_size=args.sample_size, random_seed=42, **kwargs)

    def get_tgt_loaders(batch_size=64, **kwargs):
        loader_class = DOMAIN_LOADER_MAP[target_name]
        if loader_class is None:
            raise ValueError(f"Loader for domain '{target_name}' is not implemented yet")
        loader = loader_class()
        return loader.get_loaders(batch_size=batch_size, image_size=args.target_size, load_to_memory=True, sample_size=args.sample_size, random_seed=42, **kwargs)

    print("=" * 60)
    if args.mode == 'train':
        print("DOMAIN ADVERSARIAL NEURAL NETWORK (DANN) - TRAINING")
    else:
        print("DOMAIN ADVERSARIAL NEURAL NETWORK (DANN) - TESTING")
    print("=" * 60)
    print(f"Source domain: {source_name.upper()}")
    print(f"Target domain: {target_name.upper()}")
    print(f"Sample size: {args.sample_size or 'Full dataset'}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    print(f"Batch size: {args.batch_size}")
    if args.mode == 'train':
        print(f"Learning rate: {args.lr}")
        print(f"Epochs: {args.epochs}")
        print(f"Gamma: {args.gamma}")
        print(f"Zeta: {args.zeta}")
    print(f"Save directory: {args.save_dir}")
    if args.mode == 'test':
        print(f"Model path: {args.model_path}")
    print("=" * 60)

    if args.mode == 'train':
        train_dann(
            n_epoch=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            gamma=args.gamma,
            zeta=args.zeta,
            target_size=args.target_size,
            save_dir=args.save_dir,
            source_name=source_name,
            get_src_loaders=get_src_loaders,
            target_name=target_name,
            get_tgt_loaders=get_tgt_loaders,
            input_model=CARDDModel
        )
    else:
        # Testing mode
        print("Loading model for testing...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the trained model
        model = CARDDModel()
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()

        print(f"Model loaded from: {args.model_path}")

        # Evaluate using the function from eval_dann.py
        evaluate_dann_model(model=model,
                          get_src_loaders=get_src_loaders, get_tgt_loaders=get_tgt_loaders,
                          source_name=source_name, target_name=target_name,
                          batch_size=args.batch_size, print_results=True)

