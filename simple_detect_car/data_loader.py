"""
Data loader for car scratch detection dataset.

This module provides functionality to load car scratch detection data from the
manipulated results folder, including both small samples and full datasets.
"""

import os
import json
import random
import copy
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    print("Warning: matplotlib not available. Plotting functions disabled.")

# Optional PyTorch imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features will be limited.")

# Optional torchvision transforms
if TORCH_AVAILABLE:
    try:
        from torchvision import transforms as T
        TORCHVISION_AVAILABLE = True
    except Exception:
        TORCHVISION_AVAILABLE = False
        print("Warning: torchvision not available. Data augmentation disabled.")
else:
    TORCHVISION_AVAILABLE = False


# -------------------- Transform utilities (torchvision) --------------------
def get_preprocess_transforms(target_size: Tuple[int, int] = (224, 224)):
    train_transforms = [
        T.ToPILImage(),            # accept numpy array -> PIL
        T.Resize(target_size),
        T.ToTensor(),              # PIL -> tensor in [0,1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(train_transforms)

def get_augment_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        # Note: hue parameter removed due to overflow bug in some torchvision versions
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])

def get_train_transforms(target_size: Tuple[int, int] = (224, 224)):
    preprocess_transforms = get_preprocess_transforms(target_size)
    augment_transforms = get_augment_transforms()
    return preprocess_transforms, augment_transforms

def get_eval_transforms(target_size: Tuple[int, int] = (224, 224)):
    preprocess_transforms = get_preprocess_transforms(target_size)
    return preprocess_transforms

class CarDDDataset(Dataset):
    """
    Dataset class for loading CARDD images as binary classification samples.

    Each image (both original and processed) is treated as a separate sample:
    - Original images have label 0 (no damage)
    - Processed images have label 1 (damaged)

    Returns (image_path, label) tuples.
    """
    
    def __init__(self, 
                 root: str = None,
                 domain: str = 'sd2',
                 train: bool = True,
                 split: str = None,
                 transform=None,
                 augment_transform=None,
                 sample_size: Optional[int] = None,
                 random_seed: int = 42,
                 load_to_memory: bool = True):
        """
        Initialize the CARDD dataset for binary classification.
        
        Args:
            root: Root directory containing GenAI_Results
                  (default: '../cardd_data' locally, '~/ResponsibleAI/cardd_data' on EC2)
            domain: Domain name ('sd2', 'kontext', or 'qwen')
            train: Whether to load training or validation set (deprecated, use split instead)
            split: Dataset split name ('TR', 'VAL', or 'TE'). If None, uses train parameter
            transform: Optional transform to be applied on images
            augment_transform: Optional transform to be applied on images
            sample_size: If provided, load only this many random samples
            random_seed: Random seed for reproducible sampling
            load_to_memory: If True, load all images into memory at initialization
        """
        # Set default root based on environment
        if root is None:
            # Check if we're running on EC2
            is_ec2 = (
                os.path.exists('/home/ubuntu') or  # Ubuntu EC2 default
                'ec2' in os.uname().nodename.lower() or  # hostname contains 'ec2'
                os.path.exists('/opt/aws')  # AWS tools installed
            )
            root = '~/ResponsibleAI/cardd_data' if is_ec2 else '../cardd_data'
            # Expand user path for EC2
            if is_ec2 and root.startswith('~/'):
                root = os.path.expanduser(root)

        # Set domain and train parameters
        self.domain = domain.lower()
        self.train = train  # Keep for backward compatibility
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.transform = transform
        self.augment_transform = augment_transform
        self.load_to_memory = load_to_memory
        self._images_in_memory = None  # Will store loaded images if load_to_memory=True

        # Validate domain and map to correct folder name
        domain_mapping = {
            'sd2': 'SD2',
            'kontext': 'Kontext',
            'qwen': 'Qwen Image Edit'
        }
        if self.domain not in domain_mapping:
            raise ValueError(f"Domain must be 'sd2', 'kontext', or 'qwen', got '{domain}'")

        # Set split name - support TR, VAL, or TE
        if split is not None:
            split_upper = split.upper()
            if split_upper in ['TR', 'VAL', 'TE']:
                split_name = f'CarDD-{split_upper}'
            else:
                raise ValueError(f"Split must be 'TR', 'VAL', or 'TE', got '{split}'")
        else:
            # Backward compatibility: use train parameter
            split_name = 'CarDD-TR' if train else 'CarDD-VAL'

        # Construct data and metadata directories
        self.data_dir = os.path.join(root, 'GenAI_Results', domain_mapping[self.domain], split_name)
        self.metadata_dir = os.path.join(self.data_dir, 'metadata')

        if not self._check_exists():
            raise RuntimeError(f'Dataset not found. Please ensure CARDD data is available at {self.data_dir} with metadata in {self.metadata_dir}')

        # Load samples from JSON files - each image becomes a separate sample
        self.samples = []
        import glob
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
            self.__load_dataset_to_memory()
            print("All images loaded to memory successfully.")

    def _check_exists(self):
        """Check if dataset files exist."""
        import glob
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
    
    def __len__(self) -> int:
        return len(self.samples)

    def __load_dataset_to_memory(self):
        """Load all images into memory with transform applied."""
        from tqdm import tqdm
        
        self._images_in_memory = []
        
        # Load all images
        for idx in tqdm(range(len(self.samples)), desc="Loading images to memory"):
            image_path, label = self.samples[idx]
            #  Remap paths for EC2 environment
            image_path = self._remap_path_for_ec2(image_path)
            
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply transform if provided
            if self.preprocess_transform is not None:
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
            
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply transform if provided
            if self.transform is not None:
                img = self.transform(img)
        if self.augment_transform is not None:
            img = self.augment_transform(img)

        return img, label

    def get_pair_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficiently get a random pair of original and processed images directly from metadata.

        Args:
            index: Ignored - always returns a random pair from metadata

        Returns:
            Tuple of (original_image, processed_image) as numpy arrays
        """
        import glob

        # Get all JSON files from metadata directory
        json_pattern = os.path.join(self.metadata_dir, 'processing_*.json')
        json_files = glob.glob(json_pattern)

        while True:
            json_file = random.choice(json_files)
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            if metadata.get('success', False):
                break

        # Get paths directly from metadata (most efficient approach)
        original_path = metadata.get('original_image_path', '')
        processed_path = metadata.get('processed_image_path', '')

        # Build full processed path
        if processed_path:
            processed_filename = os.path.basename(processed_path)
            full_processed_path = os.path.join(self.data_dir, processed_filename)
        else:
            full_processed_path = ''

        # Remap paths for EC2 environment
        original_path = self._remap_path_for_ec2(original_path)
        full_processed_path = self._remap_path_for_ec2(full_processed_path)

        # Load images safely
        original_img = cv2.imread(original_path, cv2.IMREAD_COLOR)
        if original_img is None:
            raise ValueError(f"Could not load image: {original_path}")
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        processed_img = cv2.imread(full_processed_path, cv2.IMREAD_COLOR)
        if processed_img is None:
            raise ValueError(f"Could not load image: {full_processed_path}")
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        return original_img, processed_img

def create_dataloader(dataset,
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = True,
                     prefetch_factor: int = 2):
    """
    Create a PyTorch DataLoader for batch training.

    Args:
        dataset: CarDDDataset instance (regular or binary classification)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (0 for single-threaded)
        pin_memory: Whether to pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        PyTorch DataLoader
    """
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available. Cannot create DataLoader.")
        return None

    # Disable pin_memory for MPS (Apple Silicon) as it's not supported
    if pin_memory and torch.backends.mps.is_available():
        pin_memory = False

    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

# Example usage and testing
if __name__ == "__main__":
    def plot_pair_demo(original_img: np.ndarray, processed_img: np.ndarray):
        """
            Demo function to plot a pair of original and processed images side by side.
        
        Args:
            original_img: Original image as numpy array
            processed_img: Processed image as numpy array
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - cannot plot images")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot original image
        ax1.imshow(original_img)
        ax1.set_title("Original Image\n(Label 0)", fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Plot processed image
        ax2.imshow(processed_img)
        ax2.set_title("Processed Image\n(Label 1)", fontsize=14, fontweight='bold')
        ax2.axis('off')

        # Main title
        plt.suptitle("Random Image Pair from CarDD Dataset",
                    fontsize=16, fontweight='bold', y=0.98)

        # Add some info text
        fig.text(0.02, 0.02, "Dataset: CarDD (Car Damage Detection)\nMethod: get_pair_image() - Efficient random sampling from metadata",
                fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

        plt.tight_layout()
        plt.show()

    # Set up paths for CARDD dataset
    base_dir = "/Users/wjs/Library/CloudStorage/OneDrive-Personal/Coding, ML & DL/ResponsibleAI/cardd_data"
    data_dir = f"{base_dir}/GenAI_Results/SD2/CarDD-TR"
    metadata_dir = f"{data_dir}/metadata"

    print("=== CarDD Dataset and DataLoader Demo ===")

    # 1. Create the CarDDDataset
    print("\n1. Creating CarDDDataset...")
    dataset = CarDDDataset(
        domain='sd2',
        train=True,
        sample_size=10,  # Use small sample for demo
        random_seed=2
    )  # Uses default root='../cardd_data'

    print(f"   Dataset size: {len(dataset)} samples")
    print("   Each sample contains: (image, label)")
    print("   - Label 0: Original image")
    print("   - Label 1: Processed image")

    # 2. Show sample data
    print("\n2. Sample data inspection:")
    for i in range(min(5, len(dataset))):
        image_path, label = dataset.samples[i]
        label_name = "Original" if label == 0 else "Processed"
        print(f"   Sample {i}: {os.path.basename(image_path)} -> Label {label} ({label_name})")

    # 3. Create DataLoader
    print("\n3. Creating DataLoader...")
    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,  # Keep order for demo
        num_workers=0
    )

    print(f"   DataLoader created with batch_size={dataloader.batch_size}")
    print(f"   Number of batches: {len(dataloader)}")

    # 5. Demonstrate get_pair_image method
    print("\n5. Demonstrating get_pair_image method...")
    if MATPLOTLIB_AVAILABLE:
        try:
            # Get a random pair of images
            original_img, processed_img = dataset.get_pair_image()

            print(f"   Successfully loaded image pair!")
            print(f"   Original image shape: {original_img.shape}")
            print(f"   Processed image shape: {processed_img.shape}")

            # Plot the images
            plot_pair_demo(original_img, processed_img)

        except Exception as e:
            print(f"   Error demonstrating image pair: {e}")
    else:
        print("   Matplotlib not available - skipping pair plotting demo")
