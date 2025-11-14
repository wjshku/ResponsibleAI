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