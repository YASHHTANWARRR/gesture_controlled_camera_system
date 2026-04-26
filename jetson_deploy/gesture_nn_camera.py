import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
import joblib

# Load model
model = load_model("gesture_nn.keras", compile=False)
label_map = joblib.load("labels.pkl")
inv_map = {v:k for k,v in label_map.items()}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

# Video recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
recording = False

prev_time = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "none"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row = np.array(row).reshape(1, -1)

            pred = model.predict(row, verbose=0)
            gesture = inv_map[np.argmax(pred)]

    # -------- EFFECTS --------

    if gesture == "zoom":
        zoom = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        frame = cv2.resize(zoom, (w, h))

    elif gesture == "blur":
        frame = cv2.GaussianBlur(frame, (25,25), 0)

    elif gesture == "edges":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif gesture == "gray":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif gesture == "highlight":
        cv2.rectangle(frame, (150,100), (450,350), (0,255,255), 3)

    # Recording control
    if gesture == "record_on":
        recording = True
    elif gesture == "record_off":
        recording = False

    if recording:
        out.write(frame)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"Gesture: {gesture}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("NN Gesture Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()