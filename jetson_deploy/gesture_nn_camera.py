import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
import joblib


recording = False
out = None

model = tf.saved_model.load("trt_model")
infer = model.signatures["serving_default"]

# labels
label_map = joblib.load("labels.pkl")
inv_map = {v: k for k, v in label_map.items()}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture("/dev/video0")
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

prev_time = 0
frame_count = 0
results = None  # for frame skipping


while True:
    success, frame = cap.read()
    if not success:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape


    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))
        print("VideoWriter opened:", out.isOpened())


    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % 2 == 0:
        results = hands.process(rgb)

    gesture = "none"


    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row = np.array(row).reshape(1, -1).astype("float32")

            # 🔥 TENSORRT INFERENCE
            pred = infer(tf.constant(row))
            pred = list(pred.values())[0].numpy()

            gesture = inv_map[np.argmax(pred)]


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


    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        recording = True
        print("Recording STARTED")

    elif key == ord('s'):
        recording = False
        print("Recording STOPPED")

    elif key == 27:
        break


    if recording and out is not None:
        out.write(frame)


    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time


    cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Jetson Optimized Gesture Camera", frame)


cap.release()

if out is not None:
    out.release()

cv2.destroyAllWindows()