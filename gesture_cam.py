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

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# 🎥 Video Writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
recording = False

prev_time = 0
gesture_history = []
SMOOTHING = 5

# YOLO frame skip (for performance)
frame_count = 0

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
    if not success:
        break

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

    # Smooth gesture
    gesture = get_smooth_gesture(gesture)

    # Zoom
    if gesture == "zoom":
        zoom = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        frame = cv2.resize(zoom, (w, h))

    # Improved background blur (center focus)
    elif gesture == "blur":
        blurred = cv2.GaussianBlur(frame, (31,31), 0)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w//2, h//2), 200, 255, -1)
        mask = cv2.GaussianBlur(mask, (51,51), 0)
        mask = mask[..., None] / 255
        frame = (frame * mask + blurred * (1 - mask)).astype(np.uint8)

    # YOLO Highlight (run every 3 frames)
    elif gesture == "highlight":
        if frame_count % 3 == 0:
            results_yolo = model(frame, verbose=False)[0]
        else:
            results_yolo = []

        if results_yolo:
            for box in results_yolo.boxes[:2]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 3)

    frame_count += 1

    # ----------- FPS -----------

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Gesture: {gesture}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    # 🎥 Recording status
    cv2.putText(frame, f"REC: {recording}", (10,110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Save video
    if recording:
        out.write(frame)

    cv2.imshow("Advanced Gesture Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'r' to toggle recording
    if key == ord('r'):
        recording = not recording

    # ESC to exit
    if key == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()