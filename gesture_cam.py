import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(1)

def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]  # index, middle, ring, pinky
    count = 0
    
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    
    return count

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    effect = "none"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = count_fingers(hand_landmarks)

            # ✋ Open palm → Zoom
            if fingers >= 4:
                effect = "zoom"

            # ✊ Fist → Blur
            elif fingers == 0:
                effect = "blur"

            # 👉 One finger → Highlight
            elif fingers == 1:
                effect = "highlight"

    # Apply effects
    if effect == "zoom":
        zoom = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        frame = cv2.resize(zoom, (w, h))
        cv2.putText(frame, "ZOOM", (10, 40), 0, 1, (0,255,0), 2)

    elif effect == "blur":
        frame = cv2.GaussianBlur(frame, (25,25), 0)
        cv2.putText(frame, "BLUR", (10, 40), 0, 1, (0,0,255), 2)

    elif effect == "highlight":
        cv2.rectangle(frame, (150,100), (450,350), (0,255,255), 3)
        cv2.putText(frame, "HIGHLIGHT", (10, 40), 0, 1, (255,255,0), 2)

    cv2.imshow("Gesture Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()