import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import mobilenet_v2


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x0, y0, x1, y1]
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2

    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    #Calculate intersection
    inter_x0 = max(x1_min, x2_min)
    inter_y0 = max(y1_min, y2_min)
    inter_x1 = min(x1_max, x2_max)
    inter_y1 = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    #Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# Feature Extractor 1 (ResNet18)
class ResNet18FeatureExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        backbone = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(backbone.children())[:-1])  # Feature block only
        self.model.eval().to(device)

        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img: Image.Image) -> np.ndarray:
        """
        Args:
            img: PIL image (already cropped ROI)
        Returns:
            np.ndarray of shape [512]
        """
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, 64, 64]
            feat = self.model(x)  # [1, 512, 1, 1]
            return feat.view(-1).cpu().numpy()  # [512]
        

# Feature Extractor 2 (MobileNet)
class MobileNetFeatureExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        model = mobilenet_v2(pretrained=True)
        self.model = model.features  # Feature block only
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.eval().to(device)

        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img: Image.Image) -> np.ndarray:
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, 64, 64]
            feat = self.model(x)  # [1, 1280, H, W]
            feat = self.pool(feat).view(-1)  # → [1280]
            return feat[:512].cpu().numpy()  # Keep first 512 dims

