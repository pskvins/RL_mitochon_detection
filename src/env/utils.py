import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import mobilenet_v2
from ultralytics import YOLO
import torch.nn.functional as F

def compute_iou(box1: np.ndarray, box2: np.ndarray, return_area: bool = False) -> float:
    """
    Compute IoU between two boxes in [x, y, w, h] format.
    """
    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2 = box2

    try:
        x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
        x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
        x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
        x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area1 = max(1e-6, w1 * h1)
        area2 = max(1e-6, w2 * h2)
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area
        if not np.isfinite(iou) or np.isnan(iou) or np.isinf(iou):
            if return_area:
                return 0.0, 0.0, area1, area2
            return 0.0
        if return_area:
            return iou, inter_area, area1, area2
        return iou
    
    except Exception as e:
        print(f"[compute_iou] Exception: {e}")
        if return_area:
            return 0.0, 0.0, 0.0, 0.0
        return 0.0


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
        
class YOLOv8FeatureExtractor:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device).eval()
        self.detect_module = None
        self.captured_features_list = []

        # Find the Detect layer
        for module in self.model.model.model:
            if module.__class__.__name__ == 'Detect':
                self.detect_module = module
                break

        if self.detect_module is None:
            raise ValueError("Could not find a Detect layer in the YOLOv8 model.")

        def hook_fn_capture(module, input_tuple, output_tuple):
            if input_tuple and isinstance(input_tuple[0], list):
                # Store all feature maps from different scales
                current_features = [f.detach().clone().cpu() for f in input_tuple[0]]
                self.captured_features_list.clear()
                self.captured_features_list.append(current_features)

        self.hook_handle = self.detect_module.register_forward_hook(hook_fn_capture)

        self.transform = T.Compose([
            T.Resize((640, 640)),  # YOLOv8 default input size
            T.ToTensor(),
        ])
    
    def __call__(self, img: Image.Image) -> np.ndarray:
        with torch.no_grad():
            # Preprocess image to YOLOv8's expected input size
            x = self.transform(img).unsqueeze(0).to(self.device)
            
            # Run prediction to trigger the hook
            _ = self.model.predict(x, verbose=False)

            if not self.captured_features_list or not self.captured_features_list[0]:
                print("[YOLOv8FeatureExtractor] WARNING: No features were captured by the hook.")
                return np.array([])

            # Get all feature maps from different scales
            feature_maps = self.captured_features_list[0]
            
            # Process each feature map
            processed_features = []
            for fm in feature_maps:
                if fm.ndim == 4:  # [1, C, H, W]
                    # Global average pooling
                    avg_pool = torch.mean(fm, dim=[-2, -1])  # [1, C]
                    processed_features.append(avg_pool)
            
            if not processed_features:
                print("[YOLOv8FeatureExtractor] WARNING: No valid feature maps to process.")
                return np.array([])

            # Concatenate features from all scales
            concatenated = torch.cat(processed_features, dim=1)  # [1, sum(C)]
            
            # Remove batch dimension and convert to numpy
            final_features = concatenated.squeeze(0).cpu().numpy()
            
            return final_features

    def __del__(self):
        if hasattr(self, 'hook_handle') and self.hook_handle:
            self.hook_handle.remove()

#get gif which present the path bbox being refined
import imageio
from PIL import Image

def make_gif(patch_seq, save_path, duration=200):
    """
    patch_seq: list of PIL.Image.Image objects
    save_path: str, path to save gif
    duration: int, duration of each frame in milliseconds
    """
    if not patch_seq:
        print("[WARNING] Empty patch sequence. GIF will not be created.")
        return
    
    frames = [p.convert("RGB") if not isinstance(p, Image.Image) else p for p in patch_seq]
    imageio.mimsave(save_path, frames, format="GIF", duration=duration / 1000.0)

class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        bias=True
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return hidden