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
        feature_extractor, max_steps=10, device="cpu"):

        self.image = image
        self.gt_boxes = gt_boxes
        self.device = device

        self.initial_box = torch.tensor(initial_box, dtype=torch.float32, device=self.device)
        self.iou_fn = iou_fn
        self.feature_extractor = feature_extractor
        self.max_steps = max_steps

    def reset(self) -> torch.Tensor:
        self.cur_box = self.initial_box.clone()
        self.step_count = 0

        if self.gt_boxes is None or len(self.gt_boxes) == 0:
            self.prev_iou = 0.0
            self.initial_iou = 0.0
        else:
            best_gt = max(self.gt_boxes, key=lambda gt: self.iou_fn(self.cur_box, gt))
            self.prev_iou = self.iou_fn(self.cur_box, best_gt)
            self.initial_iou = self.prev_iou

        return self._get_state()

    def step(self, step: int, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, dict]:
        dx, dy, dscale, p_term = action

        # === Apply shift ===
        box_w = float(self.cur_box[2])
        box_h = float(self.cur_box[3])

        dx_pix = dx * box_w
        dy_pix = dy * box_h
        self.cur_box[0] += dx_pix
        self.cur_box[1] += dy_pix

        # === Apply scale ===
        self.cur_box[2] *= 1+ dscale
        self.cur_box[3] *= 1 + dscale

        # === Clip box to image bounds ===
        img_w, img_h = self.image.size
        x, y, w, h = self.cur_box
        x1 = np.clip(x - w / 2, 0, img_w - 1)
        y1 = np.clip(y - h / 2, 0, img_h - 1)
        x2 = np.clip(x + w / 2, 0, img_w - 1)
        y2 = np.clip(y + h / 2, 0, img_h - 1)

        # 다시 center format
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        self.cur_box = torch.tensor([x, y, w, h], dtype=torch.float32, device=self.device)

        # === Done condition ===
        self.step_count += 1
        done = (self.step_count >= 3 and p_term > 0.5) or (self.step_count >= self.max_steps)

        # === Compute reward ===
        if self.gt_boxes is None or len(self.gt_boxes) == 0:
            reward = 0.0
            done = True
            return self._get_state(), reward, done, {}
        else:
            best_gt = max(self.gt_boxes, key=lambda gt: self.iou_fn(self.cur_box, gt))
            cur_iou = self.iou_fn(self.cur_box, best_gt)
            reward = 5.0 if cur_iou > self.prev_iou else -3.0

            if p_term > 0.5:
                if cur_iou > self.initial_iou:
                    reward += 6.0 + 4.0 * (10 - step) / step
                else:
                    reward -= 3.0

            self.prev_iou = cur_iou
            return self._get_state(), reward, done, {"iou": cur_iou, "step": self.step_count}

    def _get_state(self) -> torch.Tensor:
        cur_patch = self._crop_box(self.cur_box)  #PIL.Image
        feat_np = self.feature_extractor(cur_patch)  #Image -> Numpy feature
        feat = torch.tensor(feat_np, dtype=torch.float32, device=self.device)
        return torch.cat([self.cur_box, feat], dim=0)


    def _crop_box(self, box: torch.Tensor) -> Image.Image:
        """
        Return cropped image by current bbox, then resize to 64×64.
        """
        x, y, w, h = box.cpu().numpy()
        x0 = max(0, int(x - w / 2.0))
        y0 = max(0, int(y - h / 2.0))
        x1 = min(self.image.width, int(x + w / 2.0))
        y1 = min(self.image.height, int(y + h / 2.0))

        return self.image.crop((x0, y0, x1, y1)).resize((64, 64))
