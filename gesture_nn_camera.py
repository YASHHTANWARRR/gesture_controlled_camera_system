import cv2
import numpy as np
import time
from collections import Counter

cap = cv2.VideoCapture(0)

recording = False
out = None

gesture_history = []
HISTORY_SIZE = 10

last_gesture = "none"
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.medianBlur(mask, 5)

    contours_data = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_data) == 3:
        _, contours, _ = contours_data
    else:
        contours, _ = contours_data

    gesture = "none"
    finger_count = 0
    area = 0
    num_defects = 0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 20000:
            hull_indices = cv2.convexHull(largest, returnPoints=False)

            if hull_indices is not None and len(hull_indices) > 3:
                defects = cv2.convexityDefects(largest, hull_indices)

                if defects is not None:
                    num_defects = defects.shape[0]

                    for i in range(defects.shape[0]):
                        _, _, _, d = defects[i][0]
                        if d > 10000:
                            finger_count += 1

                if finger_count == 0:
                    gesture = "fist"
                elif finger_count == 1:
                    gesture = "one"
                elif finger_count == 2:
                    gesture = "two"
                elif finger_count == 3:
                    gesture = "three"
                elif finger_count >= 4:
                    gesture = "palm"

    gesture_history.append(gesture)

    if len(gesture_history) > HISTORY_SIZE:
        gesture_history.pop(0)

    most_common = Counter(gesture_history).most_common(1)[0]
    stable_gesture = most_common[0]
    confidence = most_common[1]

    current_time = time.time()

    if stable_gesture != last_gesture and (current_time - last_time) > 1.2:

        print(
            f"Gesture: {stable_gesture} | "
            f"Area: {int(area)} | "
            f"Fingers: {finger_count} | "
            f"Defects: {num_defects} | "
            f"Confidence: {confidence}/{HISTORY_SIZE}"
        )

        if stable_gesture == "palm":
            if not recording:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('gesture_record.avi', fourcc, 20.0, (640,480))
                recording = True

        elif stable_gesture == "fist":
            if recording:
                recording = False
                if out:
                    out.release()

        last_gesture = stable_gesture
        last_time = current_time

    if recording and out:
        out.write(frame)

cap.release()
if out:
    out.release()