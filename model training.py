
# 1. CHECK ENVIRONMENT

from ultralytics import YOLO
import ultralytics
ultralytics.checks()

# 2. DOWNLOAD DATASET (ROBOFLOW)

from roboflow import Roboflow

rf = Roboflow(api_key="add your api key")
project = rf.workspace("object-detection-wtxlb").project("my-first-project-sr1yz")
version = project.version(3)
dataset = version.download("yolov8")   # ✅ correct format

DATA_YAML = "D:/VS Code/practise python/My-First-Project-3/data.yaml"

# 3. TRAIN YOLO MODEL

model = YOLO("D:/VS Code/practise python/yolo11n.pt")

model.train(
    data=DATA_YAML,
    epochs=100,
    patience=20,
    imgsz=640,
    batch=-1,
    seed=47,
    freeze=10,
    dropout=0.2,
    plots=True,
    verbose=True,
    device="cpu"
)

# 4. VALIDATE MODEL

model.val(
    data=DATA_YAML,
    imgsz=640,
    save_json=True,
    plots=True
)

