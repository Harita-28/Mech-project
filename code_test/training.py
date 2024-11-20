from ultralytics import YOLO

# Path to the YAML configuration file
DATA_CONFIG_PATH = "D:/Defect_Detection/defect2/dataset/dataset.yaml"  # Ensure this path points to your dataset.yaml file

# Initialize a new YOLOv8 model (specify YOLO version: 'yolov8n', 'yolov8s', 'yolov8m', etc.)
model = YOLO('yolov8s.pt')  # Start with a pretrained model for faster training (e.g., yolov8s.pt)

# Train the model
model.train(
    data=DATA_CONFIG_PATH,  # Path to the dataset configuration file
    epochs=20,             # Number of epochs, adjust based on performance
    imgsz=640,              # Image size, can be 416, 512, 640, etc.
    batch=16,               # Batch size, adjust based on your hardware
    name='defect_detection_model',  # Name of the experiment
    project='results',  # Directory to save the results
    #device=cpu                 # Use GPU (0) if available, or "cpu" for CPU training
)
