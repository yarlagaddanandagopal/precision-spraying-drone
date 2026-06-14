import cv2
from ultralytics import YOLO

model = YOLO("best_rice_disease_model.pt")

def detect_disease():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera not accessible")
        return False

    disease_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > 0.5:
                disease_detected = True
                print("Disease Detected!")

        annotated_frame = results[0].plot()
        cv2.imshow("Live Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return disease_detected