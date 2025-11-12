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
from dann.train_dann import *
from dann.loader_dann import AbstractDANNDataset, AbstractDANNLoader, create_loader_function
from dann.model_dann import AbstractDANN
from PIL import Image
# ============================================
# DANN Model (CNN for MNIST)
# ============================================
class CNNModel(AbstractDANN):
    """
    CNN Model for DANN with MNIST/MNIST-M adaptation.

    Based on the architecture from the original DANN paper.
    Adapted for 28x28 RGB images (MNIST-M has color).

    Architecture:
        - Feature Extractor: 2-layer CNN (3→64→50 channels)
        - Label Classifier: 3-layer MLP (800→100→100→10)
        - Domain Classifier: 2-layer MLP (800→100→2)
    """

    def __init__(self):
        """
        Initialize CNN model for MNIST/MNIST-M DANN.
        """
        super(CNNModel, self).__init__()

    def _build_feature_extractor(self) -> nn.Module:
        """
        Build the feature extraction CNN backbone with flattening.

        Returns:
            Feature extractor network that outputs flattened features (batch, 800)
        """
        feature = nn.Sequential()
        feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        feature.add_module('f_bn1', nn.BatchNorm2d(64))
        feature.add_module('f_pool1', nn.MaxPool2d(2))
        feature.add_module('f_relu1', nn.ReLU(True))
        feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        feature.add_module('f_bn2', nn.BatchNorm2d(50))
        feature.add_module('f_drop1', nn.Dropout2d())
        feature.add_module('f_pool2', nn.MaxPool2d(2))
        feature.add_module('f_relu2', nn.ReLU(True))
        feature.add_module('f_flatten', nn.Flatten())  # Flatten to (batch, 50*4*4)
        return feature

    def _build_label_classifier(self) -> nn.Module:
        """
        Build the label classification head (digit classification 0-9).

        Returns:
            Label classifier network (output: 10 classes with LogSoftmax)
        """
        feature_dim = self._get_feature_dim()
        class_classifier = nn.Sequential()
        class_classifier.add_module('c_fc1', nn.Linear(feature_dim, 100))
        class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        class_classifier.add_module('c_relu1', nn.ReLU(True))
        class_classifier.add_module('c_drop1', nn.Dropout())
        class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        class_classifier.add_module('c_relu2', nn.ReLU(True))
        class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))
        return class_classifier

    def _build_domain_classifier(self) -> nn.Module:
        """
        Build the domain classification head (source vs target).

        Returns:
            Domain classifier network (output: 2 classes with LogSoftmax)
        """
        feature_dim = self._get_feature_dim()
        domain_classifier = nn.Sequential()
        domain_classifier.add_module('d_fc1', nn.Linear(feature_dim, 100))
        domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        domain_classifier.add_module('d_relu1', nn.ReLU(True))
        domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        return domain_classifier

    def _get_feature_dim(self) -> int:
        """
        Get the flattened feature dimension.

        Feature extractor output: 50 channels * 4 * 4 = 800

        Returns:
            Flattened feature dimension (800)
        """
        return 50 * 4 * 4


# ============================================
# MNIST-M Dataset (Custom dataset for MNIST-M)
# ============================================
class MNISTM(AbstractDANNDataset):
    """
    MNIST-M dataset loader.

    MNIST-M is MNIST digits blended with random patches from BSDS500 dataset.
    Images are stored as PNG files with labels in text files.
    """

    def __init__(self, root, train=True, transform=None, load_to_memory=True):
        """
        Initialize MNIST-M dataset.

        Args:
            root: Root directory containing MNIST-M data
            train: Whether to load training or test set
            transform: Optional transform to apply
            load_to_memory: If True, load all images into memory at initialization
        """
        super().__init__(root, train, transform)

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

        # Load to memory functionality
        self.load_to_memory = load_to_memory
        self._images_in_memory = None  # Will store loaded images if load_to_memory=True

        # Load all images to memory if requested
        if self.load_to_memory:
            print("Loading all MNIST-M images to memory (this may take a while)...")
            self._load_dataset_to_memory()
            print("All MNIST-M images loaded to memory successfully.")

    def _check_exists(self):
        """Check if dataset files exist."""
        train_file = self.root / 'mnist_m_train_labels.txt'
        test_file = self.root / 'mnist_m_test_labels.txt'
        train_folder = self.root / 'mnist_m_train'
        test_folder = self.root / 'mnist_m_test'
        return (train_file.exists() and test_file.exists() and
                train_folder.exists() and test_folder.exists())

    def _load_dataset_to_memory(self):
        """Load all images into memory with transform applied."""
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, desc: x  # Fallback if tqdm not available

        self._images_in_memory = []

        # Load all images
        for idx in tqdm(range(len(self.samples)), desc="Loading MNIST-M images to memory"):
            img_name, target = self.samples[idx]
            img_path = self.image_folder / img_name

            # Load image
            try:
                img = datasets.folder.pil_loader(img_path)
            except Exception as e:
                # If image loading fails, return a black PIL image
                img = Image.new('RGB', (28, 28), color='black')
                print(f"Error loading image {img_path}: {e}")

            # Apply transform if provided
            if self.transform is not None:
                img = self.transform(img)

            self._images_in_memory.append(img)

    def __getitem__(self, index):
        """Get a single item."""
        img_name, target = self.samples[index]

        # Load image from memory or disk
        if self._images_in_memory is not None and index < len(self._images_in_memory):
            # Load from memory
            img = self._images_in_memory[index]
        else:
            # Load from disk
            img_path = self.image_folder / img_name
            try:
                img = datasets.folder.pil_loader(img_path)
            except Exception as e:
                # If image loading fails, return a black PIL image
                img = Image.new('RGB', (28, 28), color='black')
                print(f"Error loading image {img_path}: {e}")

            # Apply transform if provided
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        """Return dataset size."""
        return len(self.samples)


# ============================================
# Data Loading Classes (using Abstract Base Classes)
# ============================================
class MNISTLoader(AbstractDANNLoader):
    """
    MNIST data loader class using AbstractDANNLoader.
    """

    def __init__(self, data_root: str = './data'):
        """
        Initialize MNIST loader.

        Args:
            data_root: Root directory for datasets
        """
        self.data_root = Path(data_root)

    def get_loaders(self, batch_size=64, image_size=28, load_to_memory=False, **kwargs):
        """
        Get MNIST data loaders.

        Args:
            batch_size: Batch size for data loading
            image_size: Image size (should be 28 for MNIST)
            load_to_memory: If True, preload all images into memory (torchvision already caches, so minimal benefit)
            **kwargs: Additional arguments (unused for MNIST)

        Returns:
            train_loader, test_loader: DataLoaders for MNIST
        """
        # Create transforms with grayscale to RGB conversion
        transform = self.create_default_transform(
            image_size=image_size,
            grayscale_to_rgb=True
        )

        # Load MNIST dataset (use local data if available, download otherwise)
        train_dataset = datasets.MNIST(
            root=str(self.data_root),
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root=str(self.data_root),
            train=False,
            download=True,
            transform=transform
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


class MNISTMLoader(AbstractDANNLoader):
    """
    MNIST-M data loader class using AbstractDANNLoader.
    """

    def __init__(self, data_root: str = './data'):
        """
        Initialize MNIST-M loader.

        Args:
            data_root: Root directory for datasets
        """
        self.data_root = Path(data_root)

    def get_loaders(self, batch_size=64, image_size=28, load_to_memory=True, **kwargs):
        """
        Get MNIST-M data loaders.

        Args:
            batch_size: Batch size for data loading
            image_size: Image size (should be 28 for MNIST-M)
            load_to_memory: If True, preload all images into memory at initialization
            **kwargs: Additional arguments (unused for MNIST-M)

        Returns:
            train_loader, test_loader: DataLoaders for MNIST-M
        """
        # Create transforms (MNIST-M is already RGB, no grayscale conversion needed)
        transform = self.create_default_transform(
            image_size=image_size,
            grayscale_to_rgb=False
        )

        # Load MNIST-M dataset (using local data)
        train_dataset = MNISTM(
            root=str(self.data_root / 'mnist_m'),
            train=True,
            transform=transform,
            load_to_memory=load_to_memory
        )

        test_dataset = MNISTM(
            root=str(self.data_root / 'mnist_m'),
            train=False,
            transform=transform,
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

# ============================================
# Create function-style loaders from classes
source_name, target_name = 'mnist', 'mnist_m'

def get_src_loaders(batch_size=64, **kwargs):
    loader = MNISTLoader()
    return loader.get_loaders(batch_size=batch_size, load_to_memory=False, **kwargs)  # MNIST is already cached by torchvision

def get_tgt_loaders(batch_size=64, **kwargs):
    loader = MNISTMLoader()
    return loader.get_loaders(batch_size=batch_size, load_to_memory=True, **kwargs)  # MNIST-M benefits from preloading

if __name__ == "__main__":
    train_dann(
        n_epoch=20, 
        batch_size=64, 
        save_dir="./models_mnist",
        source_name=source_name,
        get_src_loaders=get_src_loaders,
        target_name=target_name,
        get_tgt_loaders=get_tgt_loaders,
        input_model=CNNModel
    )
