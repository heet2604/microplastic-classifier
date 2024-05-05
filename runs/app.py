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

        self.output_text = None  # Initialize output_text attribute

        self.home_page()

        # Path to your custom YOLO model weights
        self.MODEL_WEIGHTS_PATH = 'C:/Users/Heet/PycharmProjects/Training/runs/detect/train4/weights/best.pt'

        # Path to the directory where you want to save the output images
        self.OUTPUT_DIR = 'C:/Users/Heet/Downloads/Microplastics'

        # Initialize your custom YOLO model
        self.model = YOLO(self.MODEL_WEIGHTS_PATH)

        # Redirect stdout and stderr to the text widget
        self.text_redirector = None  # Initialize text_redirector attribute

    def home_page(self):
        self.home_label = tk.Label(self.root, text="Welcome to Object Detection App", font=("Helvetica", 16, "bold"), bg="#222222", fg="white")
        self.home_label.pack()

        self.predict_color_button = tk.Button(self.root, text="Predict Color", command=self.predict_color, bg="#333333", fg="white")
        self.predict_color_button.pack()

        self.predict_microplastic_button = tk.Button(self.root, text="Predict Microplastic Type", command=self.predict_microplastic, bg="#333333", fg="white")
        self.predict_microplastic_button.pack()

        self.output_text = tk.Text(self.root, height=10, width=50, bg="#333333", fg="white")
        self.output_text.pack()

        # Create TextRedirector instance after output_text has been initialized
        self.text_redirector = self.TextRedirector(self.output_text)

        # Redirect stdout and stderr to the text widget
        sys.stdout = self.text_redirector
        sys.stderr = self.text_redirector

    def predict_color(self):
        self.clear_widgets()
        self.upload_image("Color")

    def predict_microplastic(self):
        self.clear_widgets()
        self.upload_image("Microplastic Type")

    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def upload_image(self, prediction_type):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.detect_objects(file_path, prediction_type)

    def detect_objects(self, image_path, prediction_type):
        # Load input image
        image = cv2.imread(image_path)

        # Perform object detection
        results = self.model(image)

        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Iterate through the list of results and save each result
        for i, result in enumerate(results):
            output_image_path = os.path.join(self.OUTPUT_DIR, f'result_{i}.jpg')
            result.save(output_image_path)

            # Display the detected image to the user
            self.display_image(output_image_path, prediction_type)

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

    class TextRedirector:
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, str):
            self.text_widget.insert("end", str)
            self.text_widget.see("end")

def main():
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
