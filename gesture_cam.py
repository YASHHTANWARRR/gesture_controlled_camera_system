import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

prev_time = 0
gesture_history = []
SMOOTHING = 5  # frames

def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]
    count = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

def get_smooth_gesture(gesture):
    gesture_history.append(gesture)
    if len(gesture_history) > SMOOTHING:
        gesture_history.pop(0)
    return max(set(gesture_history), key=gesture_history.count)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "none"

    # Hand detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(hand_landmarks)

            if fingers >= 4:
                gesture = "zoom"
            elif fingers == 0:
                gesture = "blur"
            elif fingers == 1:
                gesture = "highlight"

    # Apply smoothing
    gesture = get_smooth_gesture(gesture)

    # Zoom
    if gesture == "zoom":
        zoom = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        frame = cv2.resize(zoom, (w, h))

    # Background Blur (simple segmentation approximation)
    elif gesture == "blur":
        mask = np.zeros_like(frame)
        cv2.circle(mask, (w//2, h//2), 200, (255,255,255), -1)
        blurred = cv2.GaussianBlur(frame, (31,31), 0)
        frame = np.where(mask==255, frame, blurred)

    # Object Highlight using YOLO
    elif gesture == "highlight":
        results_yolo = model(frame, verbose=False)[0]
        for box in results_yolo.boxes[:2]:  # limit for speed
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 3)

    # ----------- FPS COUNTER -----------

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Gesture: {gesture}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Advanced Gesture Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()