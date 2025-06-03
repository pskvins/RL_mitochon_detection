import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from ultralytics import YOLO


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = box1
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
        if not np.isfinite(iou):
            return 0.0
        return iou
    
    except Exception as e:
        print(f"[compute_iou] Exception: {e}")
        return 0.0


# Feature Extractor (ResNet18)
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

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Args:
            img: PIL image (already cropped ROI)
        Returns:
            torch.Tensor of shape [512] on the correct device
        """
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, 64, 64]
            feat = self.model(x)  # [1, 512, 1, 1]
            return feat.view(-1)  # [512], torch.Tensor


class YOLOv8FeatureExtractor:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device).eval()
        self.detect_module = None
        self.captured_features_list = []

        for module in self.model.model.model:
            if module.__class__.__name__ == 'Detect':
                self.detect_module = module
                break

        if self.detect_module is None:
            raise ValueError("Could not find a Detect layer in the YOLOv8 model.")

        def hook_fn_capture(module, input_tuple, output_tuple):
            if input_tuple and isinstance(input_tuple[0], list):
                current_features = [f.detach().clone().cpu() for f in input_tuple[0]]

                self.captured_features_list.clear()
                self.captured_features_list.append(current_features)
            # The hook should not return any value that alters the normal flow
            # unless that's the specific intention (e.g., modifying the output).

        self.hook_handle = self.detect_module.register_forward_hook(hook_fn_capture)

        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
        ])

    def __call__(self, img: Image.Image) -> np.ndarray:
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0).to(self.device)

            _ = self.model.predict(x, verbose=False) 

            if not self.captured_features_list or not self.captured_features_list[0]:
                print("[YOLOv8FeatureExtractor] WARNING: No features were captured by the hook.")
                return np.array([])

            feature_maps_from_hook = self.captured_features_list[0]

            averaged_features = []
            for fm in feature_maps_from_hook:
                if fm.ndim == 4 and fm.shape[0] == 1:
                    avg_fm = torch.mean(fm, dim=[-2, -1])
                    averaged_features.append(avg_fm)
                else:
                    print(f"[YOLOv8FeatureExtractor] WARNING: Unexpected feature map shape: {fm.shape}. Skipping this map.")

            if not averaged_features:
                 print("[YOLOv8FeatureExtractor] WARNING: No valid feature maps to average after processing.")
                 return np.array([])

            concatenated_features = torch.cat(averaged_features, dim=1)

            final_feature_vector = concatenated_features.squeeze(0).cpu().numpy()

            return final_feature_vector

    def __del__(self):
        if hasattr(self, 'hook_handle') and self.hook_handle:
            self.hook_handle.remove()



#get gif which present the path bbox being refined
import imageio
from PIL import ImageDraw 
from PIL import Image, ImageDraw
import imageio
import numpy as np

def make_gif(
    base_img: Image.Image,
    coarse_boxes: list,
    refined_seqs: list,
    gt_boxes: list = None,
    save_path: str = "combined.gif",
    duration: int = 200
):
    """
    Create a GIF showing sequential refinement of boxes on a single image.

    Visual Logic:
    - GT boxes (if provided): green (always visible)
    - Initial coarse boxes: gray
    - Active refinement box: red (only one at a time)
    - Finished (refined) boxes: blue (locked after termination)

    Args:
        base_img (PIL.Image): the original image
        coarse_boxes (list): list of [x, y, w, h] initial boxes
        refined_seqs (list): list of list of [x, y, w, h] for each refinement step per box
        gt_boxes (list): optional list of GT boxes
        save_path (str): where to save the gif
        duration (int): frame duration in milliseconds
    """
    frames = []
    num_boxes = len(coarse_boxes)
    finished_boxes = []

    # Track which coarse boxes are still gray (not started yet)
    remaining_coarse_boxes = [True] * num_boxes

    for agent_idx, box_seq in enumerate(refined_seqs):
        for step_idx, step_box in enumerate(box_seq):
            frame = base_img.copy()
            draw = ImageDraw.Draw(frame)

            # Draw GT boxes (green)
            if gt_boxes:
                for gt in gt_boxes:
                    x, y, w, h = gt
                    draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="green", width=2)

            # Draw unfinished coarse boxes in gray
            for i, cb in enumerate(coarse_boxes):
                if remaining_coarse_boxes[i] and i != agent_idx:
                    x, y, w, h = cb
                    draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="gray", width=1)

            # Draw finished boxes (blue)
            for fb in finished_boxes:
                x, y, w, h = fb
                draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="blue", width=2)

            # Draw current active refinement box (red)
            x, y, w, h = step_box
            draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="red", width=2)
            draw.text((10, 10), f"Refining {agent_idx+1}/{num_boxes}, Step {step_idx+1}", fill="white")

            frames.append(frame)

        # After refinement, mark box as finished
        finished_boxes.append(step_box)
        remaining_coarse_boxes[agent_idx] = False

    # Final frame: only GT (green) and refined (blue)
    frame = base_img.copy()
    draw = ImageDraw.Draw(frame)

    if gt_boxes:
        for gt in gt_boxes:
            x, y, w, h = gt
            draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="green", width=2)

    for fb in finished_boxes:
        x, y, w, h = fb
        draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="blue", width=2)

    frames.append(frame)

    imageio.mimsave(save_path, frames, format="GIF", duration=duration)
