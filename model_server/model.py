# model.py
import sys
import os
import json
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from simple_detect_car.models import get_model
from simple_detect_car.data_loader import get_eval_transforms

class DetectionModel:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load metadata to get configuration (same as eval.py)
        model_path_obj = Path(model_path)
        metadata = None
        metadata_file = model_path_obj.parent / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = None
        
        # Get target size from metadata (default to 224x224)
        cfg = metadata.get('config', {}) if metadata else {}
        meta_target_size = tuple(cfg.get('target_size')) if cfg.get('target_size') else (224, 224)
        self.target_size = meta_target_size
        
        # Get hyperparameters from metadata
        meta_dropout = float(cfg.get('dropout', 0.5))
        meta_hidden = int(cfg.get('hidden_size', 64))
        
        # Reconstruct model with correct hyperparameters
        self.model = get_model(model_name="cnn", dropout=meta_dropout, hidden_size=meta_hidden)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Use the same eval transforms as eval.py with target size from metadata
        self.transform = get_eval_transforms(target_size=self.target_size)

    def predict(self, image: Image.Image) -> float:
        """输入一张 PIL 图像，输出车身划痕概率"""
        # The transform expects numpy array and converts to PIL, then to tensor
        # Since image is already PIL, we need to convert to numpy first
        import numpy as np
        img_array = np.array(image)
        x = self.transform(img_array).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        with torch.no_grad():
            score = self.model(x).item()
        return score