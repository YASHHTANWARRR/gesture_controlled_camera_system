# 🎥 Gesture-Controlled Camera System (Deep Learning + Jetson Nano)

A real-time **gesture-controlled camera system** using Computer Vision and Deep Learning. The system recognizes hand gestures and applies corresponding visual effects. It is designed to run on both laptops and edge devices like the NVIDIA Jetson Nano.

---

## 🚀 Features

- ✋ Real-time hand gesture recognition  
- 🧠 Deep Learning (Neural Network) based classification  
- 🎮 Gesture-controlled effects:
  - Zoom  
  - Blur  
  - Edge Detection  
  - Grayscale  
  - Object Highlight  
- 🎥 Video recording  
- 📊 FPS monitoring  
- 🐳 Docker deployment (Jetson Nano optimized)  

---

## 🧠 System Pipeline

```
Camera → MediaPipe → Hand Landmarks → Neural Network → Gesture → Effect
```

---

## 🎯 Gesture Mapping

| Gesture | Command | Effect |
|--------|--------|--------|
| ✋ Open Palm | zoom | Zoom into frame |
| ✊ Fist | blur | Background blur |
| 👉 One Finger | highlight | Highlight object |
| ✌️ Two Fingers | edges | Edge detection |
| 🤟 Three Fingers | gray | Grayscale |
| 👍 Thumb Up | record_on | Start recording |
| 👎 Thumb Down | record_off | Stop recording |

---

## 📁 Project Structure

```
gesture_controlled_camera_system/
│
├── collect_data.py
├── train_nn.py
├── gesture_nn_camera.py
│
├── gesture_nn.h5
├── labels.pkl
├── gestures.csv
│
├── Dockerfile
├── README.md
```

---

## 🧪 Setup & Installation

### 1. Create Environment

```bash
conda create -n gesture_env python=3.10
conda activate gesture_env
```

### 2. Install Dependencies

```bash
pip install opencv-python mediapipe numpy scikit-learn tensorflow joblib pandas
```

---

## 📊 Data Collection

```bash
python collect_data.py
```

Press keys:

- 1 → zoom  
- 2 → blur  
- 3 → highlight  
- 4 → edges  
- 5 → gray  

👉 Collect 100–200 samples per gesture

---

## 🧠 Train Model

```bash
python train_nn.py
```

Outputs:
- gesture_nn.h5  
- labels.pkl  

---

## 🎥 Run Application

```bash
python gesture_nn_camera.py
```

---

## 🐳 Docker Deployment (Jetson Nano)

### Build Image

```bash
sudo docker build -t gesture-jetson .
```

### Run Container

```bash
xhost +

sudo docker run -it \
--runtime nvidia \
--network host \
--privileged \
--env DISPLAY=$DISPLAY \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--device /dev/video0 \
gesture-jetson
```
## 🎬 Demo Video

Watch the real-time gesture-controlled camera system in action on the NVIDIA Jetson Nano.

### 📹 Demo Video

<video width="800" controls>
  <source src="WhatsApp Video 2026-05-21 at 9.57.17 PM.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 🎥 Demo Highlights

The demo showcases:

- ✋ Real-time hand gesture recognition  
- 🧠 Neural network-based gesture classification  
- 🎮 Dynamic visual effects:
  - Zoom
  - Blur
  - Edge Detection
  - Grayscale
  - Object Highlight
- 📊 FPS monitoring  
- 🎥 Video recording control  
- 🐳 Docker deployment on Jetson Nano  

---

## 📽️ Demo Workflow

1. Camera initialization  
2. MediaPipe hand landmark detection  
3. Gesture prediction using trained neural network  
4. Real-time effect rendering  
5. Dockerized execution on Jetson Nano edge device  

---
---

## ⚡ Optimization (Jetson Nano)

- Reduce resolution (320×240)  
- Use lightweight model  
- Skip frames  
- TensorRT (optional)  

---

## 🧠 Tech Stack

- OpenCV  
- MediaPipe  
- TensorFlow  
- Docker  

---

## 🎯 Use Cases

- Smart camera systems  
- Touchless interfaces  
- AR/VR interaction  
- Assistive technology  
- Edge AI applications  

---

## 🧠 Viva Explanation

This project uses MediaPipe to extract hand landmarks and a neural network to classify gestures. Based on predicted gestures, different image processing effects are applied in real time. The system is deployed on Jetson Nano using Docker.

---

## 🔥 Future Improvements

- TensorRT optimization  
- LSTM-based gesture recognition  
- Object detection with pointing  
- Mobile deployment  

---

## ⭐ Author

Yash Tanwar
