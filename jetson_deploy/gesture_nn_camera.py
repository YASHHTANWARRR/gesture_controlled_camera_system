import cv2
import numpy as np
import time
import tensorflow as tf
import joblib

# ---------------- INITIAL SETUP ----------------

recording = False
out = None

# TensorRT model
model = tf.saved_model.load("trt_model")
infer = model.signatures["serving_default"]

label_map = joblib.load("labels.pkl")
inv_map = {v: k for k, v in label_map.items()}

# Jetson camera
cap = cv2.VideoCapture(
    "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! appsink",
    cv2.CAP_GSTREAMER
)

if not cap.isOpened():
    cap = cv2.VideoCapture("/dev/video0")

cap.set(3, 640)
cap.set(4, 480)

prev_time = 0

# ---------------- MAIN LOOP ----------------

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))

    gesture = "none"

    # ---------------- HAND DETECTION (LIGHTWEIGHT) ----------------

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # skin color range (tune if needed)
    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(max_contour) > 3000:
            x, y, cw, ch = cv2.boundingRect(max_contour)

            cv2.rectangle(frame, (x, y), (x+cw, y+ch), (0, 255, 0), 2)

            hand_roi = frame[y:y+ch, x:x+cw]
            hand_roi = cv2.resize(hand_roi, (64, 64))

            # normalize
            hand_roi = hand_roi / 255.0
            hand_roi = hand_roi.reshape(1, -1).astype("float32")

            # TensorRT inference
            pred = infer(tf.constant(hand_roi))
            pred = list(pred.values())[0].numpy()

            gesture = inv_map[np.argmax(pred)]

    # ---------------- EFFECTS ----------------

    if gesture == "zoom":
        zoom = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        frame = cv2.resize(zoom, (w, h))

    elif gesture == "blur":
        frame = cv2.GaussianBlur(frame, (25, 25), 0)

    elif gesture == "edges":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif gesture == "gray":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif gesture == "highlight":
        cv2.rectangle(frame, (150, 100), (450, 350), (0, 255, 255), 3)

    # ---------------- RECORD CONTROL ----------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        recording = True
    elif key == ord('s'):
        recording = False
    elif key == 27:
        break

    if recording and out is not None:
        out.write(frame)

    # ---------------- FPS ----------------

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # ---------------- DISPLAY ----------------

    cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Jetson Optimized Gesture Camera", frame)

# ---------------- CLEANUP ----------------

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()