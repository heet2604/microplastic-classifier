FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install packages in specific order to avoid numpy conflicts
RUN pip install --no-cache-dir --upgrade pip

# Install numpy with specific compatible version first
RUN pip install --no-cache-dir numpy==1.23.5

# Install PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install ultralytics and other packages
RUN pip install --no-cache-dir \
    ultralytics==8.0.186 \
    opencv-python-headless==4.8.1.78 \
    streamlit==1.28.0 \
    prometheus-client==0.17.1 \
    pillow==9.5.0 \
    pandas==1.5.3

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p weights

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]