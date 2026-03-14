import cv2
from ultralytics import YOLO


MODEL_PATH = "/home/theabman/Desktop/PERSONAL PROJECTS/LineCrosser/runs/detect/runs/train/wedding_wait2/weights/best.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.4


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        annotated_frame = results[0].plot()

        cv2.imshow("Live Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()