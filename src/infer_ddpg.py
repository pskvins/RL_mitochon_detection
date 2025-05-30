import os
import torch
import numpy as np
from ultralytics import YOLO
from PIL import ImageDraw, Image
from src.env.box_env import BoxRefinementEnv
from src.env.utils import ResNet18FeatureExtractor, make_gif
from src.ddpg_model.agent import DDPGAgent

def predict(img, cfg, save_img=False, 
                       save_gif=False, 
                       save_dir="./results"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    yolo = YOLO("yolo_runs/test/finetune_exp1/weights/best.pt")
    agent = DDPGAgent(cfg['state_dim'], cfg['action_dim'], device)
    agent.load("ddpg_runs/ddpg_exp1/weights/ddpg_best.pt")
    feature_extractor = ResNet18FeatureExtractor(device=device)

    yolo_result = yolo(img, verbose = False)
    det = yolo_result[0]
    boxes = det.boxes.xywh.cpu().numpy()
    scores = det.boxes.conf.cpu().numpy()
    keep = scores >= 0.3
    coarse_boxes = boxes[keep].astype(np.float32)


    img_w, img_h = img.size
    ddpg_results = []
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)

    for i, box in enumerate(coarse_boxes):
        box = np.asarray(box, dtype=np.float32).reshape(4,)
        env = BoxRefinementEnv(
            image=img,
            gt_boxes=[],  # dummy gt box for inference
            initial_box=box,
            feature_extractor=feature_extractor,
            iou_fn=lambda a, b: 0.0  # dummy reward function for inference 
        )
        state = env.reset()
        patch_seq = []
        for _ in range(cfg.get('max_steps', 10)):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.select_action(state_tensor, noise_std=0.0)
            action_np = action.squeeze(0).cpu().numpy()
            next_state, _, done, _ = env.step(action_np)
            patch_seq.append(env._crop_box(env.cur_box))
            state = next_state
            if done:
                break

        refined_box = env.cur_box.copy()
        ddpg_results.append({"box": refined_box, "iou": None})

        x, y, w, h = refined_box
        draw.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline="red", width=2)

        if save_gif:
            gif_path = os.path.join(save_dir, f"box_{i}_refinement.gif")
            make_gif(patch_seq, gif_path)

    if save_img:
        vis_img.save(os.path.join(save_dir, "refined_result.png"))

    for idx, r in enumerate(ddpg_results):
        x, y, w, h = r["box"]
        print(f"[INFO] Refined Box {idx+1}: [x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}]")

    return ddpg_results
