from ultralytics import YOLO
import torch
import os

# Load a YOLOv8 model (you can change yolov8n.pt to yolov8s.pt or your custom model)
model = YOLO("yolov8n.pt")

# Windows-style path to your dataset YAML file
data_yaml = "C:\\Users\HP\Desktop\Final Project_model\project_model\dataset\data.yaml"

# Training parameters
img_size = 640
epochs = 50
batch_size = 16

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Train the model
results = model.train(
    data=data_yaml,
    imgsz=img_size,
    epochs=epochs,
    batch=batch_size,
    workers=4,
    device=device
)

# Define where the model gets saved after training
trained_model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
print(f"\n✅ Training complete! Model saved at: {trained_model_path}")

# Load trained model for exporting (optional)
model = YOLO(trained_model_path)

# Export to ONNX format (optional)
model.export(format="onnx")
print("📦 Model exported to ONNX format.")
