import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

def generate_coarse_boxes(
    image_dir: str,
    output_dir: str,
    model_path: str,
    conf_threshold: float = 0.3
):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])

    for path in tqdm(image_paths, desc="Running YOLO"):
        img = Image.open(path).convert("RGB")
        results = model(img, verbose=False)
        det = results[0]

        if not hasattr(det, "boxes") or det.boxes is None:
            boxes = np.empty((0, 4), dtype=np.float32)
        else:
            boxes = det.boxes.xywh.cpu().numpy()
            scores = det.boxes.conf.cpu().numpy()
            keep = scores >= conf_threshold
            boxes = boxes[keep].astype(np.float32)

        img_id = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(output_dir, img_id + ".npy")
        np.save(save_path, boxes)

if __name__ == "__main__":
    generate_coarse_boxes(
        image_dir="datasets/train/images",
        output_dir="datasets/train/coarse_boxes_yolo",
        model_path="runs/train/finetune_yolo/weights/best.pt",
        conf_threshold=0.3
    )
