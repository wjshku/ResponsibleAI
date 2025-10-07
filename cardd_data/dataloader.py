#!/usr/bin/env python3
"""
DataLoader Module

Clean and focused data loader for SOD format car damage datasets.

Author: AI Assistant
Date: 2024
"""

import os
import cv2
import numpy as np
import random
from typing import List, Dict, Any, Optional
from pathlib import Path


class DataLoader:
    """Clean and focused DataLoader for SOD format datasets"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize DataLoader
        
        Args:
            dataset_path: Path to SOD dataset root directory
        """
        self.dataset_path = dataset_path
        self._full_dataset = None
        self._dataset_size = None
        
        # Validate dataset path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
        dataset_type = dataset_path.split("/")[-1]
        # Set up image and mask directories
        self.image_dir = os.path.join(dataset_path, f"{dataset_type}-Image")
        self.mask_dir = os.path.join(dataset_path, f"{dataset_type}-Mask")
        
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"SOD directories not found: {self.image_dir}, {self.mask_dir}")
    
    @property
    def path(self) -> str:
        """Get the dataset path"""
        return self.dataset_path
    
    @property
    def size(self) -> int:
        """Get the size of the dataset"""
        if self._dataset_size is None:
            self._dataset_size = self._get_dataset_size()
        return self._dataset_size
    
    def load_full_dataset(self, shuffle: bool = True, random_seed: int = None) -> List[Dict[str, Any]]:
        """
        Load the full dataset and store it as a property
        
        Args:
            shuffle: Whether to shuffle the samples
            random_seed: Random seed for reproducible shuffling
            
        Returns:
            List of all sample dictionaries
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Get all available images
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        
        if shuffle:
            random.shuffle(image_files)
        
        samples = []
        for img_file in image_files:
            image_path = os.path.join(self.image_dir, img_file)
            mask_file = img_file.replace('.jpg', '.png')
            mask_path = os.path.join(self.mask_dir, mask_file)
            
            if os.path.exists(image_path) and os.path.exists(mask_path):
                samples.append({
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'image_file': img_file,
                    'mask_file': mask_file
                })
        
        # Store as property
        self._full_dataset = samples
        self._dataset_size = len(samples)
        
        return samples
    
    def get_sample(self, index: int = None, random_seed: int = None) -> Dict[str, Any]:
        """
        Get a sample from the full dataset
        
        Args:
            index: Specific index to get (if None, returns random sample)
            random_seed: Random seed for random sampling
            
        Returns:
            Sample dictionary
        """
        # Load full dataset if not already loaded
        if self._full_dataset is None:
            self.load_full_dataset()
        
        if not self._full_dataset:
            raise ValueError("No samples available in dataset")
        
        if index is None:
            # Return random sample
            if random_seed is not None:
                random.seed(random_seed)
            index = random.randint(0, len(self._full_dataset) - 1)
        
        if index < 0 or index >= len(self._full_dataset):
            raise IndexError(f"Index {index} out of range. Dataset size: {len(self._full_dataset)}")
        
        return self._full_dataset[index]
    
    def get_samples(self, num_samples: int, random_seed: int = None) -> List[Dict[str, Any]]:
        """
        Get multiple samples from the full dataset
        
        Args:
            num_samples: Number of samples to get
            random_seed: Random seed for random sampling
            
        Returns:
            List of sample dictionaries
        """
        # Load full dataset if not already loaded
        if self._full_dataset is None:
            self.load_full_dataset()
        
        if not self._full_dataset:
            raise ValueError("No samples available in dataset")
        
        if num_samples > len(self._full_dataset):
            raise ValueError(f"Requested {num_samples} samples, but only {len(self._full_dataset)} available")
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Get random samples
        indices = random.sample(range(len(self._full_dataset)), num_samples)
        return [self._full_dataset[i] for i in indices]
    
    def get_sample_info(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get additional information about a sample
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Dictionary with sample information
        """
        info = {
            "image_file": sample['image_file'],
            "mask_file": sample['mask_file'],
            "image_exists": os.path.exists(sample['image_path']),
            "mask_exists": os.path.exists(sample['mask_path'])
        }
        
        # Get image dimensions if image exists
        if info["image_exists"]:
            try:
                image = cv2.imread(sample['image_path'])
                if image is not None:
                    info["image_dimensions"] = image.shape[:2]
                else:
                    info["image_dimensions"] = None
            except Exception as e:
                info["image_dimensions"] = None
                info["image_load_error"] = str(e)
        
        # Get mask area if mask exists
        if info["mask_exists"]:
            try:
                mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    info["mask_area_pixels"] = int(np.sum(mask > 0))
                else:
                    info["mask_area_pixels"] = 0
            except Exception as e:
                info["mask_area_pixels"] = 0
                info["mask_load_error"] = str(e)
        
        return info
    
    def _get_dataset_size(self) -> int:
        """Get the total number of samples in the dataset"""
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        return len(image_files)
    
    def clear_cache(self):
        """Clear the cached full dataset to free memory"""
        self._full_dataset = None
        self._dataset_size = None


def visualize_results(original_path: str, mask_path: str, processed_path: str, title: str = "Processing Result"):
    """Visualize original, mask, and processed images"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Damage Mask")
    axes[1].axis('off')
    
    # Processed image
    if os.path.exists(processed_path):
        processed = cv2.imread(processed_path)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        axes[2].imshow(processed)
        axes[2].set_title("Processed Image")
    else:
        axes[2].text(0.5, 0.5, "Processed image not found", ha='center', va='center')
        axes[2].set_title("Processed Image")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    """Test cases for the clean DataLoader"""
    print("=== Clean DataLoader Test Cases ===\n")
    
    # Test 1: Initialize DataLoader
    print("1. Testing DataLoader initialization...")
    try:
        # Use default path from config
        from config import SOD_PATH
        dataset_path = os.path.join(SOD_PATH, "CarDD-TR")
        
        dataloader = DataLoader(dataset_path)
        print(f"✓ DataLoader initialized successfully")
        print(f"  Dataset path: {dataloader.path}")
        print(f"  Dataset size: {dataloader.size}")
    except Exception as e:
        print(f"✗ Failed to initialize DataLoader: {e}")
        return
    
    print()
    
    # Test 2: Load full dataset
    print("2. Testing full dataset loading...")
    try:
        full_dataset = dataloader.load_full_dataset(shuffle=True, random_seed=42)
        print(f"✓ Successfully loaded full dataset")
        print(f"  Number of samples: {len(full_dataset)}")
        print(f"  Dataset size property: {dataloader.size}")
        
        # Show first sample
        if full_dataset:
            first_sample = full_dataset[0]
            print(f"  First sample: {first_sample['image_file']}")
    except Exception as e:
        print(f"✗ Full dataset loading failed: {e}")
        return
    
    print()
    
    # Test 3: Get individual samples
    print("3. Testing sample retrieval...")
    try:
        # Get random sample
        random_sample = dataloader.get_sample(random_seed=42)
        print(f"✓ Got random sample: {random_sample['image_file']}")
        
        # Get specific sample
        specific_sample = dataloader.get_sample(index=0)
        print(f"✓ Got specific sample (index 0): {specific_sample['image_file']}")
        
        # Get multiple samples
        multiple_samples = dataloader.get_samples(num_samples=3, random_seed=42)
        print(f"✓ Got {len(multiple_samples)} samples")
        for i, sample in enumerate(multiple_samples):
            print(f"  Sample {i+1}: {sample['image_file']}")
        
    except Exception as e:
        print(f"✗ Sample retrieval failed: {e}")
    
    print()
    
    # Test 4: Test sample info
    print("4. Testing sample info extraction...")
    try:
        sample = dataloader.get_sample(index=0)
        sample_info = dataloader.get_sample_info(sample)
        
        print(f"✓ Sample info extracted successfully")
        print(f"  Image file: {sample_info['image_file']}")
        print(f"  Mask file: {sample_info['mask_file']}")
        print(f"  Image exists: {sample_info['image_exists']}")
        print(f"  Mask exists: {sample_info['mask_exists']}")
        if 'image_dimensions' in sample_info and sample_info['image_dimensions']:
            print(f"  Image dimensions: {sample_info['image_dimensions']}")
        if 'mask_area_pixels' in sample_info:
            print(f"  Mask area: {sample_info['mask_area_pixels']} pixels")
        
    except Exception as e:
        print(f"✗ Sample info extraction failed: {e}")
    
    print()
    
    
    print("\n=== Clean DataLoader Test Complete ===")


if __name__ == "__main__":
    main()