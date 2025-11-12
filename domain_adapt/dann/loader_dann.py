#!/usr/bin/env python3
"""
Abstract base classes for DANN data loaders.

This module provides abstract base classes that define the interface for creating
data loaders for DANN training. Subclass these to create new data loaders for
different datasets.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# ============================================
# Abstract Base Class for Custom Datasets
# ============================================
class AbstractDANNDataset(Dataset, ABC):
    """
    Abstract base class for custom DANN datasets.
    
    Subclass this to create custom datasets for DANN training.
    All datasets should return (image, label) tuples.
    """
    
    def __init__(self, root: str, train: bool = True, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory containing dataset files
            train: Whether to load training or test set
            transform: Optional transform to apply to images
        """
        self.root = Path(root)
        self.train = train
        self.transform = transform
        
    @abstractmethod
    def _check_exists(self) -> bool:
        """
        Check if dataset files exist.
        
        Returns:
            True if dataset files are available, False otherwise
        """
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple:
        """
        Get a single item from the dataset.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Tuple of (image, label) where:
            - image: Transformed image tensor
            - label: Integer label (0-9 for digit classification)
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the size of the dataset.
        
        Returns:
            Number of samples in the dataset
        """
        pass


# ============================================
# Abstract Base Class for Data Loader Functions
# ============================================
class AbstractDANNLoader(ABC):
    """
    Abstract base class for DANN data loader functions.
    
    This class defines the interface that all data loader functions must follow.
    Subclass this to create new data loaders for different datasets.
    
    Example:
        class MyDatasetLoader(AbstractDANNLoader):
            def get_loaders(self, batch_size=64, image_size=28, **kwargs):
                # Implementation here
                return train_loader, test_loader
    """
    
    @abstractmethod
    def get_loaders(self, 
                    batch_size: int = 64, 
                    image_size: int = 28,
                    **kwargs) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and test data loaders for a dataset.
        
        Args:
            batch_size: Batch size for data loading
            image_size: Target image size (height/width)
            **kwargs: Additional dataset-specific arguments
            
        Returns:
            Tuple of (train_loader, test_loader) where both are PyTorch DataLoaders
            
        Raises:
            RuntimeError: If dataset files are not found
            ValueError: If invalid parameters are provided
        """
        pass
    
    def create_default_transform(self, image_size: int = 28,
                                normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                                normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                                grayscale_to_rgb: bool = False) -> transforms.Compose:
        """
        Create a default transform pipeline for image preprocessing.

        Args:
            image_size: Target image size (will be converted to (image_size, image_size))
            normalize_mean: Mean values for normalization (RGB channels)
            normalize_std: Std values for normalization (RGB channels)
            grayscale_to_rgb: Whether to convert grayscale to RGB (3 channels)

        Returns:
            Composed transform pipeline
        """
        transform_list = [
            transforms.Resize((image_size, image_size)),  # Ensure square output size
        ]
        
        if grayscale_to_rgb:
            transform_list.append(transforms.Grayscale(3))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        
        return transforms.Compose(transform_list)
    
    def create_dataloader(self, 
                         dataset: Dataset,
                         batch_size: int = 64,
                         shuffle: bool = True,
                         num_workers: int = 2) -> DataLoader:
        """
        Create a PyTorch DataLoader from a dataset.
        
        Args:
            dataset: PyTorch Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            
        Returns:
            PyTorch DataLoader instance
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )


# ============================================
# Convenience Function Wrapper
# ============================================
def create_loader_function(loader_class: type, **init_kwargs) -> callable:
    """
    Create a function-style loader from a loader class.
    
    This allows using loader classes as functions for backward compatibility.
    
    Args:
        loader_class: Subclass of AbstractDANNLoader
        **init_kwargs: Keyword arguments to pass to loader class __init__
        
    Returns:
        Function that creates an instance and calls get_loaders()
        
    Example:
        class MyLoader(AbstractDANNLoader):
            def __init__(self, data_root='./data'):
                self.data_root = data_root
            def get_loaders(self, batch_size=64, image_size=224):  # Custom default
                # Implementation
                return train_loader, test_loader

        # With default data_root and loader's default image_size
        get_my_loaders = create_loader_function(MyLoader)
        train_loader, test_loader = get_my_loaders(batch_size=32)

        # With custom data_root and loader's default image_size
        get_my_loaders_custom = create_loader_function(MyLoader, data_root='./custom_data')
        train_loader, test_loader = get_my_loaders_custom(batch_size=32)

        # Override image_size
        train_loader, test_loader = get_my_loaders(batch_size=32, image_size=128)
    """
    def loader_function(batch_size=64, **kwargs):
        loader_instance = loader_class(**init_kwargs)
        # Only pass image_size if explicitly provided, otherwise let loader use its default
        loader_kwargs = {'batch_size': batch_size, **kwargs}
        return loader_instance.get_loaders(**loader_kwargs)
    return loader_function


# ============================================
# Example Implementation Template
# ============================================
class ExampleDANNLoader(AbstractDANNLoader):
    """
    Example implementation template for creating new data loaders.
    
    Copy this class and modify to create your own data loader.
    """
    
    def __init__(self, data_root: str = './data', dataset_name: str = 'example'):
        """
        Initialize the loader.
        
        Args:
            data_root: Root directory for datasets
            dataset_name: Name of the dataset
        """
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
    
    def get_loaders(self, 
                    batch_size: int = 64, 
                    image_size: int = 28,
                    **kwargs) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and test data loaders.
        
        This is a template - replace with your actual implementation.
        """
        # Create transforms
        transform = self.create_default_transform(
            image_size=image_size,
            grayscale_to_rgb=kwargs.get('grayscale_to_rgb', False)
        )
        
        # Create datasets (replace with your actual dataset class)
        # train_dataset = YourDataset(root=str(self.data_root / self.dataset_name), 
        #                             train=True, transform=transform)
        # test_dataset = YourDataset(root=str(self.data_root / self.dataset_name), 
        #                            train=False, transform=transform)
        
        # Create data loaders
        # train_loader = self.create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = self.create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # return train_loader, test_loader
        
        raise NotImplementedError("Subclass ExampleDANNLoader and implement get_loaders()")


if __name__ == "__main__":
    # Example usage
    print("Abstract DANN Loader Base Classes")
    print("=" * 50)
    print("\nTo create a new loader:")
    print("1. Subclass AbstractDANNLoader")
    print("2. Implement get_loaders() method")
    print("3. Use create_loader_function() for function-style interface")
    print("\nExample:")
    print("""
    class MyDatasetLoader(AbstractDANNLoader):
        def get_loaders(self, batch_size=64, image_size=28, **kwargs):
            transform = self.create_default_transform(image_size=image_size)
            # ... create datasets and loaders ...
            return train_loader, test_loader
    
    get_my_loaders = create_loader_function(MyDatasetLoader)
    train_loader, test_loader = get_my_loaders(batch_size=32)
    """)
