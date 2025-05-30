import numpy as np
from typing import Tuple
from PIL import Image
import torch

class BoxRefinementEnv:
    """
    RL environment for single-box refinement.
    """
    def __init__(
        self, image: Image.Image, gt_boxes: np.ndarray,
        initial_box: np.ndarray, iou_fn,
        feature_extractor, max_steps= 10):

        """
        Args:
            image: PIL image
            gt_boxes: numpy array of GT boxes [N, 4]
            initial_box: numpy array [x, y, w, h]
            iou_fn: function(box1, box2) -> float

            feature_extractor: function(img_crop) -> feature vector (np.ndarray)
            feature_dim: dimension of feature extracted by feature_extractor

            max_steps: maximum steps per episode
        """
        self.image = image
        self.gt_boxes = gt_boxes
        self.initial_box = np.array(initial_box, dtype=np.float32).reshape(-1)
        self.iou_fn = iou_fn

        self.feature_extractor = feature_extractor
        self.max_steps = max_steps

    def reset(self) -> np.ndarray:
        """
        reset environment.
        """
        self.cur_box = self.initial_box.copy()
        self.step_count = 0
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        dx, dy, dscale, p_term = action

        # Apply shift
        self.cur_box[0] += dx
        self.cur_box[1] += dy

        # Apply scale
        scale = max(1.0 + dscale, 0.1)  # prevent collapse
        self.cur_box[2] *= scale
        self.cur_box[3] *= scale

        # Compute reward
        best_gt = max(self.gt_boxes, key=lambda gt: self.iou_fn(self.cur_box, gt))
        iou = self.iou_fn(self.cur_box, best_gt)

        # Done condition
        self.step_count += 1
        done = (p_term > 0.5) or (self.step_count >= self.max_steps)

        return self._get_state(), iou, done, {}

    def _get_state(self) -> np.ndarray:
        """
        Return state which is concatenated vector of
            CNN feature and current position of box
        """
        cur_patch = self._crop_box(self.cur_box)
        feat = self.feature_extractor(cur_patch)
        return np.concatenate([self.cur_box, feat], axis=0)

    def _crop_box(self, box: np.ndarray) -> Image.Image:
        """
        Return cropped image by current bbox
        Args:
            box: current bbox 
        """
        x, y, w, h = box
        x0 = int(x - w / 2)
        y0 = int(y - h / 2)
        x1 = int(x + w / 2)
        y1 = int(y + h / 2)
        return self.image.crop((x0, y0, x1, y1)).resize((64, 64))
