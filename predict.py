import os
from ultralytics import YOLO
import cv2

# Path to your custom YOLO model weights
MODEL_WEIGHTS_PATH = 'C:/Users/Heet/PycharmProjects/Training/runs/detect/train4/weights/best.pt'

# Path to your input image
INPUT_IMAGE_PATH = "C:/Users/Heet/AppData/Local/Temp/efa8e2fc-421a-48e0-baab-a786fc658e00_Microplastics.v1i.yolov8 (1).zip.e00/train/images/a01-20-_jpg.rf.a6891b55570e9b81fd1226d04081e9b9.jpg# Path to the directory where you want to save the output image"
OUTPUT_DIR =  'C:/Users/Heet/Downloads/Microplastics'

# Load input image
image = cv2.imread(INPUT_IMAGE_PATH)

# Initialize your custom YOLO model
model = YOLO(MODEL_WEIGHTS_PATH)

# Perform object detection
results = model(image)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Iterate through the list of results and save each result
for i, result in enumerate(results):
    output_image_path = os.path.join(OUTPUT_DIR, f'result_{i}.jpg')
    result.show()
    # Save the output image
    result.save(output_image_path)

# Release resources
cv2.destroyAllWindows()
