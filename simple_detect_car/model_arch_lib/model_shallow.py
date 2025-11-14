import torch.nn as nn

class CNNClassifier(nn.Module):
    """
    Shallow CNN Model for car detection.

    Architecture adapted for car detection:
    - Feature Extractor: ResNet-like CNN backbone
    - Label Classifier: Binary classification (car/not car)
    """

    def __init__(self, num_classes=1):
        """
        Initialize shallow CNN model.

        Args:
            num_classes: Number of label classes (default: 1 for binary classification)
            input_size: Input image size (default: 224)
        """
        super(CNNClassifier, self).__init__()
        self.num_classes = num_classes

        # Build feature extractor
        self.feature_extractor = self._build_feature_extractor()

        # Build label classifier
        self.label_classifier = self._build_label_classifier()

    def _build_feature_extractor(self) -> nn.Module:
        """
        Build the feature extraction CNN backbone with flattening.

        Uses a ResNet-like architecture adapted for car detection.

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

        Feature extractor output: 256 channels * 4 * 4 = 4096

        Returns:
            Flattened feature dimension (4096)
        """
        return 256 * 4 * 4

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
