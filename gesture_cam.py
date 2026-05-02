import cv2
import numpy as np

# Try loading YOLO (optional)
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    use_yolo = True
except:
    print("YOLO not available, running basic camera only")
    use_yolo = False

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not opening")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize for performance
    frame = cv2.resize(frame, (640, 480))

    if use_yolo:
        try:
            results = model(frame)
            annotated = results[0].plot()
            cv2.imshow("Gesture Camera (YOLO)", annotated)
        except:
            cv2.imshow("Gesture Camera", frame)
    else:
        cv2.imshow("Gesture Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()