import torch.nn as nn

# ============================================
# Basic Residual Block for ResNet
# ============================================
class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ============================================
# Deep ResNet-18 Model for Car Detection
# ============================================
class CNNClassifier(nn.Module):
    """
    ResNet-18 Model for car detection.

    Architecture adapted for car detection:
    - Feature Extractor: ResNet-18 backbone (standard architecture)
    - Label Classifier: Binary classification (car/not car)
    """

    def __init__(self, num_classes=1):
        """
        Initialize ResNet-18 model for car detection.

        Args:
            num_classes: Number of label classes (default: 1 for binary classification)
        """
        super(CNNClassifier, self).__init__()
        self.num_classes = num_classes

        # Build feature extractor
        self.feature_extractor = self._build_feature_extractor()

        # Build label classifier
        self.label_classifier = self._build_label_classifier()

    def _build_feature_extractor(self) -> nn.Module:
        """
        Build the ResNet-18 feature extraction backbone.

        Standard ResNet-18 architecture:
        - Conv1: 7x7 conv, 64 channels, stride 2
        - MaxPool: 3x3, stride 2
        - Layer1: 2 residual blocks, 64 channels
        - Layer2: 2 residual blocks, 128 channels
        - Layer3: 2 residual blocks, 256 channels
        - Layer4: 2 residual blocks, 512 channels
        - AdaptiveAvgPool + Flatten

        Total: 18 convolutional layers (standard ResNet-18)

        Returns:
            Feature extractor network that outputs flattened features
        """
        # Initial convolution and pooling
        layers = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]

        # Layer 1: 2 residual blocks, 64 channels
        self._make_layer(layers, 64, 64, num_blocks=2, stride=1)

        # Layer 2: 2 residual blocks, 128 channels
        self._make_layer(layers, 64, 128, num_blocks=2, stride=2)

        # Layer 3: 2 residual blocks, 256 channels
        self._make_layer(layers, 128, 256, num_blocks=2, stride=2)

        # Layer 4: 2 residual blocks, 512 channels
        self._make_layer(layers, 256, 512, num_blocks=2, stride=2)

        # Adaptive pooling and flattening
        layers.extend([
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        ])

        return nn.Sequential(*layers)

    def _make_layer(self, layers, in_channels, out_channels, num_blocks, stride):
        """Helper method to create residual layers."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # First block with potential downsampling
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

    def _build_label_classifier(self) -> nn.Module:
        """
        Build the label classification head (car detection: car/not car).

        Returns:
            Label classifier network (output: num_classes with Sigmoid)
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
        class_classifier.add_module('sigmoid', nn.Sigmoid())
        return class_classifier

    def _get_feature_dim(self) -> int:
        """
        Get the flattened feature dimension.

        ResNet-18 feature extractor output: 512 channels * 4 * 4 = 8192

        Returns:
            Flattened feature dimension (8192)
        """
        return 512 * 4 * 4

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Classification output
        """
        # Feature extraction
        features = self.feature_extractor(x)

        # Classification
        output = self.label_classifier(features)

        return output
