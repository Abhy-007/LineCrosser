import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "/home/theabman/Desktop/PERSONAL PROJECTS/LineCrosser/runs/detect/runs/train/wedding_wait5/weights/best.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.4
TRACKER_CONFIG = "bytetrack.yaml"
CROSSING_COOLDOWN = 1.0  # seconds

track_history = {}
last_cross_time = {}
last_event_text = ""


def get_side_of_line(cx, line_x):
    if cx < line_x:
        return "left"
    elif cx > line_x:
        return "right"
    return "on_line"


def main():
    global last_event_text

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    names = model.names
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        frame_h, frame_w = frame.shape[:2]

        # Vertical line at center of frame
        LINE_X = frame_w // 2

        # Draw vertical line
        cv2.line(frame, (LINE_X, 0), (LINE_X, frame_h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            "Crossing Line",
            (LINE_X + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        results = model.track(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None:
            boxes = result.boxes
            xyxy_list = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
            conf_list = boxes.conf.cpu().tolist() if boxes.conf is not None else []
            cls_list = boxes.cls.cpu().tolist() if boxes.cls is not None else []
            id_list = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(xyxy_list)

            for xyxy, conf, cls_id, track_id in zip(xyxy_list, conf_list, cls_list, id_list):
                if track_id == -1:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                cls_id = int(cls_id)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                label = f"{names[cls_id]} | ID {track_id} | {conf:.2f}"

                # Draw box and center point
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                current_side = get_side_of_line(cx, LINE_X)

                if track_id in track_history:
                    prev_cx, prev_cy = track_history[track_id]
                    prev_side = get_side_of_line(prev_cx, LINE_X)

                    enough_time_passed = (
                        track_id not in last_cross_time or
                        now - last_cross_time[track_id] > CROSSING_COOLDOWN
                    )

                    if enough_time_passed:
                        if prev_side == "left" and current_side == "right":
                            last_event_text = f"{names[cls_id]} | ID {track_id} EXITED"
                            last_cross_time[track_id] = now
                            print(last_event_text)

                        elif prev_side == "right" and current_side == "left":
                            last_event_text = f"{names[cls_id]} | ID {track_id} ENTERED"
                            last_cross_time[track_id] = now
                            print(last_event_text)

                track_history[track_id] = (cx, cy)

        if last_event_text:
            cv2.putText(
                frame,
                last_event_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow("Live Line Crossing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()