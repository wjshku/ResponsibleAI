"""
Data loader for car scratch detection dataset.

This module provides functionality to load car scratch detection data from the
manipulated results folder, including both small samples and full datasets.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

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
def get_default_transforms(target_size: Tuple[int, int] = (224, 224), augment: bool = True):
    """Return a torchvision Compose for training/eval.

    - When augment=True: include flips, rotation, color jitter.
    - Always converts to tensor and normalizes to ImageNet stats.
    """
    if not TORCHVISION_AVAILABLE:
        return None

    train_transforms = [
        T.ToPILImage(),            # accept numpy array -> PIL
        T.Resize(target_size),
    ]
    if augment:
        train_transforms.extend([
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])
    train_transforms.extend([
        T.ToTensor(),              # PIL -> tensor in [0,1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return T.Compose(train_transforms)


def get_eval_transforms(target_size=(224, 224)):
    return get_default_transforms(target_size=target_size, augment=False)


class CarScratchDataset:
    """
    Dataset class for car scratch detection.
    
    This dataset loads original images, masks, and processed (fake) images
    from the car scratch manipulation results.
    """
    
    def __init__(self, 
                 data_dir: str,
                 metadata_dir: str,
                 sample_size: Optional[int] = None,
                 random_seed: int = 42,
                 transform=None,
                 load_masks: bool = True,
                 load_processed: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the directory containing processed images
            metadata_dir: Path to the directory containing metadata JSON files
            sample_size: If provided, load only this many random samples. If None, load all data.
            random_seed: Random seed for reproducible sampling
            transform: Optional transform to be applied on images
            load_masks: Whether to load mask images
            load_processed: Whether to load processed (fake) images
        """
        self.data_dir = Path(data_dir)
        self.metadata_dir = Path(metadata_dir)
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.transform = transform
        self.load_masks = load_masks
        self.load_processed = load_processed
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter successful entries
        all_valid_entries = [
            entry for entry in self.metadata 
            if entry.get('success', False) and 
            self._check_file_exists(entry)
        ]
        
        # Apply sampling if requested
        if sample_size is not None and sample_size < len(all_valid_entries):
            random.seed(random_seed)
            self.valid_entries = random.sample(all_valid_entries, sample_size)
            print(f"Loaded {len(self.valid_entries)} sample entries from {len(all_valid_entries)} total valid entries")
        else:
            self.valid_entries = all_valid_entries
            print(f"Loaded {len(self.valid_entries)} valid entries from {len(self.metadata)} total entries")
    
    def _load_metadata(self) -> List[Dict]:
        """Load all metadata from JSON files."""
        metadata = []
        
        # Load processing batch metadata files
        for json_file in self.metadata_dir.glob("processing_*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    metadata.append(data)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
        
        return metadata
    
    def _check_file_exists(self, entry: Dict) -> bool:
        """Check if all required files exist for an entry."""
        # Check original image
        if not os.path.exists(entry['original_image_path']):
            return False
        
        # Check mask if needed
        if self.load_masks and not os.path.exists(entry['mask_path']):
            return False
        
        # Check processed image if needed
        if self.load_processed:
            processed_path = entry['processed_image_path']
            # Extract just the filename from the path
            processed_filename = os.path.basename(processed_path)
            # Look for the file in data_dir
            processed_path = self.data_dir / processed_filename
            if not os.path.exists(processed_path):
                return False
        
        return True
    
    def __len__(self) -> int:
        return len(self.valid_entries)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, str, Dict]]:
        """Get a single data item."""
        entry = self.valid_entries[idx]
        
        # Load original image
        original_img = self._load_image(entry['original_image_path'])
        
        result = {
            'image_id': entry['image_id'],
            'original_image': original_img,
            'metadata': entry
        }
        
        # Load mask if requested
        if self.load_masks:
            mask = self._load_image(entry['mask_path'], grayscale=True)
            result['mask'] = mask
        
        # Load processed image if requested
        if self.load_processed:
            processed_path = entry['processed_image_path']
            # Extract just the filename from the path
            processed_filename = os.path.basename(processed_path)
            # Look for the file in data_dir
            processed_path = self.data_dir / processed_filename
            processed_img = self._load_image(str(processed_path))
            result['processed_image'] = processed_img
        
        return result
    
    def _load_image(self, path: str, grayscale: bool = False) -> np.ndarray:
        """Load an image from file path."""
        try:
            if grayscale:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            
            return img
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a dummy image if loading fails
            if grayscale:
                return np.zeros((100, 100), dtype=np.uint8)
            else:
                return np.zeros((100, 100, 3), dtype=np.uint8)
    
    @classmethod
    def load_binary_dataset(cls, 
                           data_dir: str, 
                           metadata_dir: str, 
                           sample_size: Optional[int] = None,
                           shuffle: bool = True,
                           random_seed: int = 42,
                           transform=None,
                           forbidden_original_paths: Optional[set] = None,
                           forbidden_processed_paths: Optional[set] = None) -> 'CarScratchDataset':
        """
        Load dataset for binary classification (real vs processed).
        
        Each sample contains:
        - x: image array (original or processed)
        - y: label (0 for real/original, 1 for processed/fake)
        
        Args:
            data_dir: Path to the directory containing processed images
            metadata_dir: Path to the directory containing metadata JSON files
            sample_size: If provided, load only this many random samples. If None, load all data.
            shuffle: Whether to shuffle the data
            random_seed: Random seed for reproducible sampling
            transform: Optional transform to be applied on images
            
        Returns:
            CarScratchDataset configured for binary classification
        """
        class BinaryClassificationDataset(cls):
            def __init__(self, *args, **kwargs):
                # Pop leakage-control kwargs before calling base __init__
                f_orig = kwargs.pop('forbidden_original_paths', None)
                f_proc = kwargs.pop('forbidden_processed_paths', None)
                super().__init__(*args, **kwargs)
                # Create shuffled indices for proper shuffling
                if shuffle:
                    random.seed(random_seed)
                    self.shuffled_indices = list(range(len(self.valid_entries)))
                    random.shuffle(self.shuffled_indices)
                else:
                    self.shuffled_indices = list(range(len(self.valid_entries)))
                # Leakage control: forbid sampling of certain originals/processed
                self.forbidden_original_paths = set(f_orig) if f_orig else set()
                self.forbidden_processed_paths = set(f_proc) if f_proc else set()
                # Cache to avoid repeated string lookups per index across epochs
                self._forbid_cache = {}
            
            def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
                """Get a single data item for binary classification."""
                # Use shuffled index if shuffling is enabled
                actual_idx = self.shuffled_indices[idx]
                
                # Get the base data
                base_data = super().__getitem__(actual_idx)
                # Lazy-compute forbidden flags for this index once and cache
                cached = self._forbid_cache.get(actual_idx)
                if cached is None:
                    entry_meta = base_data.get('metadata', {})
                    orig_path_meta = str(entry_meta.get('original_image_path', ''))
                    proc_path_meta = str(entry_meta.get('processed_image_path', ''))
                    forbid_orig = bool(orig_path_meta and orig_path_meta in self.forbidden_original_paths)
                    forbid_proc = bool(proc_path_meta and proc_path_meta in self.forbidden_processed_paths)
                    cached = (forbid_orig, forbid_proc)
                    self._forbid_cache[actual_idx] = cached
                forbid_orig, forbid_proc = cached
                
                # Randomly choose between original (label 0) and processed (label 1)
                want_original = random.random() < 0.5
                image = None
                label = None
                # Enforce forbidden lists to avoid leakage between splits
                if want_original and forbid_orig:
                    want_original = False  # force processed
                elif (not want_original) and forbid_proc:
                    want_original = True  # force original

                if want_original and 'original_image' in base_data:
                    image = base_data['original_image']
                    label = 0
                else:
                    # Fallback to processed if original unavailable or forbidden
                    image = base_data.get('processed_image', base_data.get('original_image'))
                    label = 1 if 'processed_image' in base_data else 0
                
                # Apply transforms if provided
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        return BinaryClassificationDataset(
            data_dir=data_dir,
            metadata_dir=metadata_dir,
            sample_size=sample_size,
            random_seed=random_seed,
            transform=transform,
            load_masks=False,  # Don't need masks for classification
            load_processed=True,  # Need processed images for labels
            forbidden_original_paths=forbidden_original_paths,
            forbidden_processed_paths=forbidden_processed_paths
        )


def create_train_test_split(dataset, test_size=0.2, random_state=42):
    """Create train/test split for binary classification dataset."""
    import random
    random.seed(random_state)
    
    # Get all valid indices
    all_indices = list(range(len(dataset.valid_entries)))
    random.shuffle(all_indices)
    
    # Split indices
    split_idx = int(len(all_indices) * (1 - test_size))
    train_indices = all_indices[:split_idx]
    test_indices = all_indices[split_idx:]
    
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    return train_indices, test_indices


def create_dataloader(dataset, 
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = True,
                     target_size: Tuple[int, int] = (224, 224)):
    """
    Create a PyTorch DataLoader for batch training.
    
    Args:
        dataset: CarScratchDataset instance (regular or binary classification)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (0 for single-threaded)
        pin_memory: Whether to pin memory for faster GPU transfer
        target_size: Target size for image resizing (height, width)
        
    Returns:
        PyTorch DataLoader
    """
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available. Cannot create DataLoader.")
        return None
    
    # Create a wrapper that resizes images to the same size
    class ResizeWrapper:
        def __init__(self, dataset, target_size):
            self.dataset = dataset
            self.target_size = target_size
        
        def __len__(self):
            return len(self.dataset)
        
        def _to_numpy_hwc_uint8(self, img):
            """Convert possible torch tensor to numpy HWC uint8 in [0,255]."""
            try:
                import torch as _torch
            except Exception:
                _torch = None
            if _torch is not None and isinstance(img, _torch.Tensor):
                # Accept (C,H,W) or (H,W,C)
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    # (C,H,W) -> (H,W,C)
                    img = img.permute(1, 2, 0)
                # Detach and move to cpu
                img = img.detach().cpu().numpy()
                # If normalized (roughly [-2,2]) or [0,1], bring to [0,255]
                # Heuristic: if max <= 1.5, assume [0,1]
                maxv = float(np.nanmax(img)) if np.isfinite(img).all() else 1.0
                minv = float(np.nanmin(img)) if np.isfinite(img).all() else 0.0
                if maxv <= 1.5 and minv >= -0.5:
                    img = (img * 255.0).clip(0, 255)
                else:
                    # If seemingly normalized by ImageNet stats, roughly shift to 0..1 not attempted
                    # Just clip to [0,255] after scaling if needed
                    pass
                img = img.astype(np.uint8)
            return img

        def __getitem__(self, idx):
            item = self.dataset[idx]
            
            # If it's a binary classification dataset (returns tuple)
            if isinstance(item, tuple):
                image, label = item
                # If tensor, keep as tensor and resize with torch (preserve normalization)
                try:
                    import torch as _torch
                    import torch.nn.functional as _F
                except Exception:
                    _torch = None
                if _torch is not None and isinstance(image, _torch.Tensor):
                    # Expect (C,H,W); add batch dim and interpolate
                    if image.ndim == 3 and image.shape[0] in (1, 3):
                        image_b = image.unsqueeze(0)
                        image_r = _F.interpolate(image_b, size=self.target_size, mode='bilinear', align_corners=False)
                        image = image_r.squeeze(0)
                    else:
                        # Fallback to numpy path if unexpected shape
                        img_np = self._to_numpy_hwc_uint8(image)
                        image = cv2.resize(img_np, (self.target_size[1], self.target_size[0]))
                else:
                    img_np = self._to_numpy_hwc_uint8(image)
                    image = cv2.resize(img_np, (self.target_size[1], self.target_size[0]))
                return image, label
            else:
                # If it's a regular dataset (returns dict)
                result = item.copy()
                try:
                    import torch as _torch
                    import torch.nn.functional as _F
                except Exception:
                    _torch = None
                for k in ('original_image', 'processed_image', 'mask'):
                    if k in result and result[k] is not None:
                        v = result[k]
                        if _torch is not None and isinstance(v, _torch.Tensor):
                            if v.ndim == 3 and v.shape[0] in (1, 3):
                                vb = v.unsqueeze(0)
                                vr = _F.interpolate(vb, size=self.target_size, mode='bilinear', align_corners=False)
                                result[k] = vr.squeeze(0)
                            else:
                                img_np = self._to_numpy_hwc_uint8(v)
                                result[k] = cv2.resize(img_np, (self.target_size[1], self.target_size[0]))
                        else:
                            img_np = self._to_numpy_hwc_uint8(v)
                            result[k] = cv2.resize(img_np, (self.target_size[1], self.target_size[0]))
                return result
    
    # Wrap the dataset with resizing
    resized_dataset = ResizeWrapper(dataset, target_size)
    
    return DataLoader(
        resized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def get_dataset_info(metadata_dir: str) -> Dict:
    """
    Get information about the dataset.
    
    Args:
        metadata_dir: Path to the metadata directory
        
    Returns:
        Dictionary containing dataset statistics
    """
    metadata_dir = Path(metadata_dir)
    metadata = []
    for json_file in metadata_dir.glob("processing_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                metadata.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    # Calculate statistics
    total_entries = len(metadata)
    successful_entries = sum(1 for entry in metadata if entry.get('success', False))
    
    # Get image dimensions
    dimensions = [entry.get('image_dimensions', [0, 0]) for entry in metadata if entry.get('success', False)]
    widths = [d[0] for d in dimensions if len(d) >= 2]
    heights = [d[1] for d in dimensions if len(d) >= 2]
    
    # Get mask areas
    mask_areas = [entry.get('mask_area_pixels', 0) for entry in metadata if entry.get('success', False)]
    
    # Get prompts
    prompts = [entry.get('tool_parameters', {}).get('prompt', '') for entry in metadata if entry.get('success', False)]
    unique_prompts = list(set(prompts))
    
    return {
        'total_entries': total_entries,
        'successful_entries': successful_entries,
        'success_rate': successful_entries / total_entries if total_entries > 0 else 0,
        'image_dimensions': {
            'widths': widths,
            'heights': heights,
            'avg_width': np.mean(widths) if widths else 0,
            'avg_height': np.mean(heights) if heights else 0,
            'min_width': min(widths) if widths else 0,
            'max_width': max(widths) if widths else 0,
            'min_height': min(heights) if heights else 0,
            'max_height': max(heights) if heights else 0,
        },
        'mask_areas': {
            'areas': mask_areas,
            'avg_area': np.mean(mask_areas) if mask_areas else 0,
            'min_area': min(mask_areas) if mask_areas else 0,
            'max_area': max(mask_areas) if mask_areas else 0,
        },
        'unique_prompts': unique_prompts,
        'num_unique_prompts': len(unique_prompts)
    }


# Example usage and testing
if __name__ == "__main__":
    # Set up paths
    base_dir = "/Users/wjs/Library/CloudStorage/OneDrive-Personal/Coding, ML & DL/ResponsibleAI/cardd_data"
    data_dir = f"{base_dir}/manipulated_results"
    metadata_dir = f"{base_dir}/manipulated_results/metadata"

    # Load full dataset
    print("\nLoading full dataset...")
    full_dataset = CarScratchDataset(
        data_dir=data_dir,
        metadata_dir=metadata_dir
    )
    print(f"Full dataset size: {len(full_dataset)}")
    
    # Test binary classification dataset
    print("\nTesting binary classification dataset...")
    binary_dataset = CarScratchDataset.load_binary_dataset(
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        sample_size=10,
        shuffle=True,
        random_seed=42
    )
    print(f"Binary dataset size: {len(binary_dataset)}")
    
    # Test a few samples
    for i in range(min(3, len(binary_dataset))):
        image, label = binary_dataset[i]
        label_name = "REAL" if label == 0 else "FAKE"
        print(f"Sample {i}: Image shape {image.shape}, Label: {label} ({label_name})")
    
    # Test DataLoader for batch training
    print("\nTesting DataLoader for batch training...")
    dataloader = create_dataloader(
        binary_dataset, 
        batch_size=2, 
        shuffle=True
    )
    
    if dataloader:
        print(f"DataLoader created successfully")
        print(f"Number of batches: {len(dataloader)}")
        
        # Test one batch
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels: {labels.tolist()}")
            break  # Only show first batch
    else:
        print("PyTorch not available, skipping DataLoader test")

    # ------------------------------------------------------------------
    # Demo: using get_default_transforms / get_eval_transforms
    # ------------------------------------------------------------------
    if TORCHVISION_AVAILABLE:
        print("\nDemonstrating transforms usage...")
        from pprint import pprint

        # Training-time transforms with augmentation (applied to individual images)
        train_transform = get_default_transforms(target_size=(224, 224), augment=True)
        # Evaluation-time transforms (no augmentation)
        eval_transform = get_eval_transforms(target_size=(224, 224))

        # Build small datasets to visualize output types/shapes
        train_demo_ds = CarScratchDataset.load_binary_dataset(
            data_dir=data_dir,
            metadata_dir=metadata_dir,
            sample_size=2,
            shuffle=True,
            transform=train_transform,
        )

        eval_demo_ds = CarScratchDataset.load_binary_dataset(
            data_dir=data_dir,
            metadata_dir=metadata_dir,
            sample_size=2,
            shuffle=False,
            transform=eval_transform,
        )

        from utils import visualize_transforms

        print("\nTrain-time transforms (with augmentation):")
        for i in range(min(2, len(train_demo_ds))):
            image, label = train_demo_ds[i]
            print(f"Image shape: {tuple(image.shape)}, Label: {label}")
            visualize_transforms(image, f"Augmented training image {i}")

    else:
        print("\ntorchvision not available; skipping transforms demo.")