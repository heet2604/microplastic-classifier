import tkinter as tk
from tkinter import filedialog
import cv2
import os
from ultralytics import YOLO
from PIL import Image, ImageTk
import sys


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")
        self.root.configure(bg="#222222")  # Set background color to dark gray

        self.output_text = None
        self.total_objects_label = None  # Label to display the total number of objects detected

        self.home_page()

        # Path to your custom YOLO model weights for microplastic type prediction
        self.MODEL_WEIGHTS_PATH = 'C:/Users/Heet/PycharmProjects/Training/runs/detect/train4/weights/best.pt'

        # Path to your custom YOLO model weights for color prediction
        self.MODEL_COLOR_WEIGHTS_PATH = 'C:/Users/Heet/OneDrive/Documents/color_best.pt'

        # Path to the directory where you want to save the output images
        self.OUTPUT_DIR = 'C:/Users/Heet/Downloads/Microplastics'

        # Initialize your custom YOLO model
        self.model = YOLO(self.MODEL_WEIGHTS_PATH)

    def home_page(self):
        self.home_label = tk.Label(self.root, text="Microplastic Classifier", font=("Arial", 24, "bold italic"),
                                   bg="#222222", fg="white")
        self.home_label.pack(pady=(50, 20))  # Add padding above and below the heading

        # Style for the buttons
        button_style = {"bg": "#333333", "fg": "white", "font": ("Arial", 12), "relief": "raised", "borderwidth": 2}

        self.predict_color_button = tk.Button(self.root, text="Predict Color", command=self.predict_color,
                                              **button_style)
        self.predict_color_button.pack(pady=(0, 10))  # Add padding below the "Predict Color" button

        self.predict_microplastic_button = tk.Button(self.root, text="Predict Microplastic Type",
                                                     command=self.predict_microplastic, **button_style)
        self.predict_microplastic_button.pack(pady=10)  # Add padding below the "Predict Microplastic Type" button

        self.output_text = tk.Text(self.root, height=10, width=50, bg="#333333", fg="white")
        self.output_text.pack()

    def predict_color(self):
        self.clear_widgets()
        self.upload_image("Color", self.MODEL_COLOR_WEIGHTS_PATH)  # Use different YOLO model for color prediction

    def predict_microplastic(self):
        self.clear_widgets()
        self.upload_image("Microplastic Type", self.MODEL_WEIGHTS_PATH)

    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def upload_image(self, prediction_type, model_weights_path):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.detect_objects(file_path, prediction_type, model_weights_path)

    def detect_objects(self, image_path, prediction_type, model_weights_path):
        # Load input image
        image = cv2.imread(image_path)

        # Initialize YOLO model with the specified weights
        model = YOLO(model_weights_path)

        # Perform object detection
        results = model(image)

        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Iterate through the list of results and save each result
        for i, result in enumerate(results):
            output_image_path = os.path.join(self.OUTPUT_DIR, f'result_{prediction_type}_{i}.jpg')
            result.save(output_image_path)

            # Display the detected image to the user
            self.display_image(output_image_path, prediction_type)

            # Update the total number of objects label
            self.count_objects(results)

    def display_image(self, image_path, prediction_type):
        # Load the image using PIL
        image = Image.open(image_path)
        image = image.resize((400, 400))  # Resize the image if needed

        # Convert the image to Tkinter format
        photo = ImageTk.PhotoImage(image)

        # Create a label to display the image
        image_label = tk.Label(self.root, image=photo, bg="#222222")
        image_label.image = photo  # Keep a reference to prevent garbage collection
        image_label.pack()

        self.home_button = tk.Button(self.root, text="Home", command=self.home_page, bg="#333333", fg="white")
        self.home_button.pack()

    def count_objects(self, results):
        # Count the total number of objects detected
        total_objects = sum(len(result) for result in results)

        # Update the total number of objects label
        self.update_total_objects_label(total_objects)

    def update_total_objects_label(self, total_objects):
        if self.total_objects_label:
            self.total_objects_label.destroy()

        self.total_objects_label = tk.Label(self.root, text=f"Total Objects Detected: {total_objects}",
                                            bg="#222222", fg="white")
        self.total_objects_label.pack()


def main():
    root = tk.Tk()

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the x and y coordinates for the Tk root window
    x = (screen_width // 2) - (800 // 2)  # Center the window horizontally and set width to 800
    y = (screen_height // 2) - (600 // 2)  # Center the window vertically and set height to 600

    # Set the geometry of the window
    root.geometry(f"800x600+{x}+{y}")

    app = ObjectDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
