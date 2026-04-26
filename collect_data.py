import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

def save_landmarks(hand_landmarks, label):
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])

    with open("gestures.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row + [label])

print("Press keys to record:")
print("1=zoom, 2=blur, 3=highlight, 4=edges, 5=gray")

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('1'):
                save_landmarks(hand_landmarks, "zoom")
            elif key == ord('2'):
                save_landmarks(hand_landmarks, "blur")
            elif key == ord('3'):
                save_landmarks(hand_landmarks, "highlight")
            elif key == ord('4'):
                save_landmarks(hand_landmarks, "edges")
            elif key == ord('5'):
                save_landmarks(hand_landmarks, "gray")

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()