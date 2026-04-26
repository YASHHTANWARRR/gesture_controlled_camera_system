FROM nvcr.io/nvidia/l4t-ml:r35.3.1-py3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy joblib

COPY . .

CMD ["python3", "gesture_nn_camera.py"]