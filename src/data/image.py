import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel
import requests
from src.data.dataset import compute_iou
import torchvision.ops as ops
import torch.nn.functional as F

def apply_deltas(box, deltas):
    x,y,w,h=box
    deltas = deltas.squeeze(0)  # Ensure deltas is a 1D tensor
    dx,dy,dw,dh = deltas
    pred_xc = x + dx * w
    pred_yc = y + dy * h
    pred_w = w * torch.exp(dw)
    pred_h = h * torch.exp(dh)

    return torch.stack([pred_xc, pred_yc, pred_w, pred_h])

def xywh_to_xyxy(box):
    """
    Convert bounding box from [xc, yc, w, h] format to [x1, y1, x2, y2] format.
    """
    xc, yc, w, h = box
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return torch.stack([x1, y1, x2, y2])

def l1_score(pred, target):
    pred = xywh_to_xyxy(pred)  # Convert to [x1, y1, x2, y2]
    target = xywh_to_xyxy(target)  # Convert to [x1, y1, x2, y2]
    return -F.l1_loss(pred, target, reduction="none").sum(dim=-1)

def giou(box1, box2):
    box1 = xywh_to_xyxy(box1)  # Convert to [x1, y1, x2, y2]
    box2 = xywh_to_xyxy(box2)  # Convert to [x1, y1, x2, y2]
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)
    giou = ops.generalized_box_iou(box1, box2)  # [N, N] → assume diagonal
    loss = 1 - torch.diag(giou)
    return loss


def dpo_loss_fn(policy_pred_boxes, ref_pred_boxes, chosen_targets, rejected_targets, beta, box_loss_type='giou'):
    """
    DPO loss function for training the policy model.
    
    Args:
        policy_pred_boxes (torch.Tensor): Predicted boxes from the policy model.
        ref_pred_boxes (torch.Tensor): Predicted boxes from the reference model.
        chosen_targets (torch.Tensor): Ground truth boxes for "good" examples.
        rejected_targets (torch.Tensor): Ground truth boxes for "bad" examples.
        beta (float): Scaling factor for the loss.
        box_loss_type (str): Type of box loss to use ('iou' or 'l1').
    
    Returns:
        torch.Tensor: Computed DPO loss.
    """
    if box_loss_type == 'giou':
        score_policy_chosen = -giou(policy_pred_boxes, chosen_targets)
        score_policy_rejected = -giou(policy_pred_boxes, rejected_targets)
        score_ref_chosen = -giou(ref_pred_boxes, chosen_targets)
        score_ref_rejected = -giou(ref_pred_boxes, rejected_targets)
    elif box_loss_type == 'l1':
        score_policy_chosen = l1_score(policy_pred_boxes, chosen_targets)
        score_policy_rejected = l1_score(policy_pred_boxes, rejected_targets)
        score_ref_chosen = l1_score(ref_pred_boxes, chosen_targets)
        score_ref_rejected = l1_score(ref_pred_boxes, rejected_targets)

    else:
        raise ValueError(f"Unsupported box loss type: {box_loss_type}")
    #print("policy_pred_boxes:", policy_pred_boxes)
    #print("ref_pred_boxes:", ref_pred_boxes)
    #print("chosen_targets:", chosen_targets)
    #print("rejected_targets:", rejected_targets)
    #print("GIoU(policy, chosen):", score_policy_chosen)
    #print("GIoU(policy, rejected):", score_policy_rejected)
    epsilon=0.1
    score_policy_chosen += epsilon * torch.randn_like(score_policy_chosen)
    score_policy_rejected += epsilon * torch.randn_like(score_policy_rejected)
    pi_logratios = score_policy_chosen - score_policy_rejected  # How much policy prefers chosen over rejected
    ref_logratios = score_ref_chosen - score_ref_rejected # How much reference prefers chosen over rejected
    #logits = beta * (pi_logratios - ref_logratios)
    logits = beta*pi_logratios
    if torch.isnan(logits).any():
        print("NaN detected in DPO logits. Clamping may be needed or check scores.")
        print("score_policy_chosen:", score_policy_chosen)
        print("score_policy_rejected:", score_policy_rejected)
        print("score_ref_chosen:", score_ref_chosen)
        print("score_ref_rejected:", score_ref_rejected)

    loss = -F.logsigmoid(logits).mean()  # DPO loss is -log(sigmoid(logits))
    return loss

class FeatureExtractor(nn.Module):
    """
        feature extractor for image patch corresponeding to coarse box
    """ 
    def __init__(self, device : str = "cuda"):
        super().__init__()
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.patch_size = self. model.config.patch_size
        self.model.eval().to(device)
        self.feature_dim = 768  # DINOv2 base model feature dimension
        self.output_dim = 4
        self.transform = T.Resize((224, 224))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # For pooling the patch features to a fixed size
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim + 256, self.feature_dim//2),
            nn.ReLU(),
            nn.Linear(self.feature_dim//2, self.feature_dim//4),
            nn.ReLU(),
            nn.Linear(self.feature_dim//4, self.output_dim)
        )
        self.fc.to(device)
        self.input_proj = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.input_proj.to(device)

    def __call__(self, img : Image.Image, input_box : np.ndarray) -> torch.Tensor:
        cropped = self.crop_box_from_image(img, input_box)
        cropped = self.transform(cropped)
        inputs = self.processor(images=cropped, return_tensors="pt").to(self.device)
        batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
        num_patches_height, num_patches_width = img_height // self.patch_size, img_width // self.patch_size
        outputs = self.model(**inputs)
        patch_feat = outputs.last_hidden_state[:,1:,:].unflatten(1, (num_patches_height, num_patches_width))
        patch_feat = patch_feat.permute(0,3,2,1)
        x = self.avgpool(patch_feat)
        x = torch.flatten(x, 1)

        # input_box 형태 바꿔야함. (crop 했을 때에 위치.?)
        input_edited = self.change_box(input_box, img.size)
        input_edited = input_edited.unsqueeze(0)
        input_edited = self.input_proj(input_edited)
        combined_feat = torch.cat((x, input_edited), dim=1)

        deltas = self.fc(combined_feat)

        return deltas
    

    def denomralize_box_expansion(self, box_norm, img_size: tuple) -> torch.Tensor:
        """
        Denormalize a box from normalized coordinates to pixel coordinates.
        Args:
            box_norm (np.ndarray): Normalized box coordinates [x, y, w, h].
            img_size (tuple): Size of the image (width, height).
        Returns:
            torch.Tensor: Denormalized box coordinates in pixel space.
        """
        x, y, w, h = box_norm
        width, height = img_size
        x_denorm = x * width
        y_denorm = y * height
        w_denorm = w * width
        h_denorm = h * height
        x1 = x_denorm - w_denorm *0.75
        y1 = y_denorm - h_denorm *0.75
        x2 = x_denorm + w_denorm *0.75
        y2 = y_denorm + h_denorm *0.75
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
    
    def crop_box_from_image(self, img, box_norm):
        """
        Crop a box from an image using normalized coordinates.
        Args:
            img (Image.Image): Input image.
            box_norm (np.ndarray): Normalized box coordinates [x, y, w, h].
        Returns:
            Image.Image: Cropped image patch.
        """
        x1, y1, x2, y2  = self.denomralize_box_expansion(box_norm, img.size)
        img_w, img_h = img.size
        x1 = float(max(0, min(x1, img_w - 1)))
        y1 = float(max(0, min(y1, img_h - 1)))
        x2 = float(max(0, min(x2, img_w - 1)))
        y2 = float(max(0, min(y2, img_h - 1)))
        cropped = img.crop((x1, y1, x2, y2))
        return cropped

    def change_box(self, box_norm, img_size: tuple) -> torch.Tensor:
        """
        Change the box format from [x, y, w, h] to [x',y', w',h'] 
        [x',y','w',h'] in cropped image
        """
        x,y,w,h = box_norm
        width, height = img_size
        x_denorm = x * width
        y_denorm = y * height
        w_denorm = w * width
        h_denorm = h * height

        # Calculate the crop box
        crop_x1, crop_y1, crop_x2, crop_y2 = self.denomralize_box_expansion(box_norm, img_size)
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        # Normalize the box coordinates to the cropped image
        x_new = (x_denorm - crop_x1) / crop_w
        y_new = (y_denorm - crop_y1) / crop_h
        w_new = w_denorm / crop_w
        h_new = h_denorm / crop_h

        return torch.tensor([x_new, y_new, w_new, h_new], dtype=torch.float32).to(self.device)