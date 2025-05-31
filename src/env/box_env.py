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
        feature_extractor, max_steps= 10, device = "cpu"):

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
            self.initial_iou = self.prev_iou  # save initial IoU of coarse boxes

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, dict]:
        dx, dy, dscale, p_term = action

        # Apply shift
        self.cur_box[0] += dx
        self.cur_box[1] += dy

        # Apply scale
        scale = np.clip(1.0 + dscale, 0.5, 2.0) #prevent bbox collapse or explosion     
        self.cur_box[2] *= scale
        self.cur_box[3] *= scale

        #clip box to image bound
        img_w, img_h = self.image.size
        x, y, w, h = self.cur_box
        w = min(w, img_w)
        h = min(h, img_h)
        x = np.clip(x, w / 2, img_w - w / 2)
        y = np.clip(y, h / 2, img_h - h / 2)
        self.cur_box = torch.tensor([x, y, w, h], dtype=torch.float32, device=self.device)  # ✅

        # Done condition
        self.step_count += 1
        done = (p_term > 0.5) or (self.step_count >= self.max_steps)

        # Compute Rewards
        # 1. use delta_IoU = cur_iou - prev_iou
        # 2. clipping by tanh
        # 3. give a small penalty for action (to minimize number of step)
        # 4. give extra reward/penalty for termination
        if self.gt_boxes is None or len(self.gt_boxes) == 0: 
            reward = 0.0
            done = True
            return self._get_state(), reward, done, {}
        else:
            best_gt = max(self.gt_boxes, key=lambda gt: self.iou_fn(self.cur_box, gt))
            cur_iou = self.iou_fn(self.cur_box, best_gt)

            iou_delta = cur_iou - self.prev_iou
            reward = np.tanh(iou_delta) * 100 #scaling reward

            if p_term > 0.5: #when the agent select to terminate
                if cur_iou > self.initial_iou: #refinement success
                    reward += 1.0
                else: #refinement failed
                    reward -= 1.0



        return self._get_state(), reward, done, {"iou": cur_iou,
                                                 "step": self.step_count}

    def _get_state(self) -> torch.Tensor:
        """
        Return state which is concatenated vector of
            CNN feature and current position of box
        """
        cur_patch = self._crop_box(self.cur_box)
        feat = self.feature_extractor(cur_patch).to(self.device)
        return torch.cat([self.cur_box, feat], axis=0)

    def _crop_box(self, box: np.ndarray) -> Image.Image:
        """
        Return cropped image by current bbox
        Args:
            box: current bbox 
        """
        x, y, w, h = box
        x0 = max(0, int(x - w / 2))
        y0 = max(0, int(y - h / 2))
        x1 = min(self.image.width, int(x + w / 2))
        y1 = min(self.image.height, int(y + h / 2))

        return self.image.crop((x0, y0, x1, y1)).resize((64, 64))
