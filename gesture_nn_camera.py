import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
import joblib

# ---------------- INITIAL SETUP ----------------

recording = False   # recording starts OFF
out = None          # video writer (created after first frame)

# load trained model + labels
model = load_model("gesture_nn.keras", compile=False)
label_map = joblib.load("labels.pkl")
inv_map = {v: k for k, v in label_map.items()}

# mediapipe hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# start webcam
cap = cv2.VideoCapture(0)

prev_time = 0  # for FPS

# ---------------- MAIN LOOP ----------------

while True:
    success, frame = cap.read()

    # if camera fails, exit
    if not success:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)  # mirror view
    h, w, _ = frame.shape

    # create video writer only once (after we know frame size)
    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))
        
        # check if writer is working
        print("VideoWriter opened:", out.isOpened())

    # convert to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "none"

    # ---------------- HAND DETECTION ----------------

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # draw hand skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # extract landmark values (x, y, z)
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row = np.array(row).reshape(1, -1)

            # predict gesture
            pred = model.predict(row, verbose=0)
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

    # gesture-based (may not trigger if not trained)
    if gesture == "record_on":
        recording = True
        print("Recording STARTED (gesture)")

    elif gesture == "record_off":
        recording = False
        print("Recording STOPPED (gesture)")

    # keyboard control (ALWAYS works)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        recording = True
        print("Recording STARTED (keyboard)")

    elif key == ord('s'):
        recording = False
        print("Recording STOPPED (keyboard)")

    elif key == 27:  # ESC
        break

    # ---------------- WRITE VIDEO ----------------

    if recording and out is not None:
        print("Writing frame...")  # debug
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

    cv2.imshow("NN Gesture Camera", frame)

# ---------------- CLEANUP ----------------

cap.release()

# release video file properly (very important)
if out is not None:
    out.release()

cv2.destroyAllWindows()