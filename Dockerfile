FROM nvcr.io/nvidia/l4t-ml:r35.3.1-py3

WORKDIR /app

# install basic dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip3 install numpy==1.24.3 joblib mediapipe==0.10.9 tensorflow==2.13.0

# copy project files
COPY . .

# run app
CMD ["python3", "gesture_nn_camera.py"]