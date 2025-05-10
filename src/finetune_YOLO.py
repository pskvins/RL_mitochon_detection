from ultralytics import YOLO
from data.prepare_dataset import prepare_dataset
from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parents[1])

model = YOLO("yolov8n.pt")

yaml_path = prepare_dataset(
    zip_path="data/datasets.zip",
    extract_to="datasets"
)

model.train(
    data=yaml_path,
    epochs=100,
    save_period=10,
    project="runs/train",
    name="finetune_exp1"
)
