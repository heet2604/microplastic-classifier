# Python In-built packages
from pathlib import Path
import PIL
import time

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Prometheus Metrics (Conditional import)
try:
    from prometheus_client import start_http_server, Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Prometheus client not available, continuing without metrics")

# Initialize Prometheus metrics
if PROMETHEUS_AVAILABLE:
    try:
        start_http_server(8000)
        REQUEST_COUNT = Counter('app_requests_total', 'Total app requests')
        DETECTION_COUNT = Counter('detections_total', 'Total microplastic detections')
        REQUEST_DURATION = Histogram('app_request_duration_seconds', 'Request latency')
        ACTIVE_USERS = Gauge('app_active_users', 'Active users')
        APP_START_TIME = Gauge('app_start_time', 'Application start time')
        APP_START_TIME.set_to_current_time()
    except Exception as e:
        print(f"Failed to start Prometheus server: {e}")
        PROMETHEUS_AVAILABLE = False

# Track detection function
def track_detection(count=1):
    if PROMETHEUS_AVAILABLE:
        DETECTION_COUNT.inc(count)

def track_request():
    if PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.inc()

# Setting page layout
st.set_page_config(
    page_title="Microplastic detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Microplastic detection using YOLOv8")

# Show Prometheus status
if PROMETHEUS_AVAILABLE:
    st.sidebar.success("✅ Monitoring enabled (Port 8000)")
else:
    st.sidebar.warning("⚠️ Monitoring disabled")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['color', 'Type'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Track sidebar interactions
if PROMETHEUS_AVAILABLE:
    ACTIVE_USERS.inc()

# Selecting Detection Or Segmentation
if model_type == 'color':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Type':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None

# Initialize total detected objects count
total_detected_objects_count = 0

# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image, caption="Microplastics on Land",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(uploaded_image, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image, caption='Microplastics In water',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                # Track the request
                track_request()
                
                start_time = time.time()
                res = model.predict(uploaded_image, conf=confidence)
                detection_time = time.time() - start_time
                
                if PROMETHEUS_AVAILABLE:
                    REQUEST_DURATION.observe(detection_time)
                
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")
                
                # Track detections and update count
                detection_count = len(boxes)
                total_detected_objects_count += detection_count
                track_detection(detection_count)
                
                # Show detection metrics
                st.sidebar.metric("Detection Time", f"{detection_time:.2f}s")
                st.sidebar.metric("Objects Detected", detection_count)

# Display the total number of detected objects
st.sidebar.header("Detection Summary")
st.sidebar.write(f"**Total objects detected:** {total_detected_objects_count}")

if source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

# Add a simple health check endpoint for Docker
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# This won't interfere with Streamlit but provides a health endpoint
if __name__ == "__main__":
    # This is just for documentation - Streamlit handles the main execution
    pass