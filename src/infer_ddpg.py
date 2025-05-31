import os
import torch
import numpy as np
from ultralytics import YOLO
from PIL import ImageDraw, Image
from src.env.box_env import BoxRefinementEnv
from src.env.utils import ResNet18FeatureExtractor, make_gif
from src.ddpg_model.agent import DDPGAgent

def predict(img, save_img=True, save_gif=False, save_dir="./results"):   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    yolo = YOLO("yolo_runs/train/finetune_exp1/weights/best.pt")
    agent = DDPGAgent(516, 4, device)
    agent.load("ddpg_epoch_100.pt", map_location=device)
    feature_extractor = ResNet18FeatureExtractor(device=device)

    yolo_result = yolo(img, verbose=False)
    det = yolo_result[0]
    boxes = det.boxes.xywh  # Tensor [N, 4]
    scores = det.boxes.conf
    keep = scores >= 0.3
    coarse_boxes = boxes[keep].to(device).float()  # Tensor [M, 4]

    ddpg_results = []
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)
    refined_seq = []

    for i, box in enumerate(coarse_boxes):
        box_seq = []
        env = BoxRefinementEnv(
            image=img,
            gt_boxes=[],  # no gt in inference
            initial_box=box,
            feature_extractor=feature_extractor,
            iou_fn=lambda a, b: 0.0,  # dummy iou
            device=device
        )
        state = env.reset()

        for _ in range(10):
            state_tensor = state.unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.select_action(state_tensor, noise_std=0.0)
            action_np = action.squeeze(0)
            next_state, _, done, _ = env.step(action_np)
            box_seq.append(env.cur_box.clone())
            state = next_state
            # if done: break

        refined_seq.append(box_seq)
        refined_box = env.cur_box.clone()
        ddpg_results.append({"box": refined_box, "iou": None})

        x, y, w, h = refined_box.tolist()
        draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline="blue", width=2)

    if save_gif:
        gif_path = os.path.join(save_dir, f"refinement.gif")
        make_gif(img, coarse_boxes.tolist(), refined_seq, save_path=gif_path)

    if save_img:
        vis_img.save(os.path.join(save_dir, "refined_result.png"))

    for idx, r in enumerate(ddpg_results):
        x, y, w, h = r["box"].tolist()
        print(f"[INFO] Refined Box {idx+1}: [x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}]")

    return ddpg_results
