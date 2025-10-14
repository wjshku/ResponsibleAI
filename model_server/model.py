# model.py
import sys
import os
import torch
from torchvision import transforms
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from detection_car.models import get_model

class ScratchModel:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(model_name="cnn", dropout=0.5, hidden_size=64)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 统一图像预处理（保证输入尺寸正确）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, image: Image.Image) -> float:
        """输入一张 PIL 图像，输出车身划痕概率"""
        x = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        with torch.no_grad():
            score = self.model(x).item()
        return score