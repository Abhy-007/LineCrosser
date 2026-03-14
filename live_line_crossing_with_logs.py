import cv2
import time
import csv
from ultralytics import YOLO

MODEL_PATH = "/home/theabman/Desktop/PERSONAL PROJECTS/LineCrosser/runs/detect/runs/train/wedding_wait5/weights/best.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.4
TRACKER_CONFIG = "bytetrack.yaml"

ZONE_WIDTH = 60  # pixels around the line

track_history = {}
object_state = {}

CSV_FILE = "crossing_log.csv"


def get_side(cx, line_x):
    if cx < line_x:
        return "left"
    elif cx > line_x:
        return "right"
    return "on_line"


def log_event(timestamp, object_id, cls_name, direction, dwell_time):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, object_id, cls_name, direction, dwell_time])


def main():
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

        frame_h, frame_w = frame.shape[:2]

        LINE_X = frame_w // 2
        LEFT_ZONE = LINE_X - ZONE_WIDTH // 2
        RIGHT_ZONE = LINE_X + ZONE_WIDTH // 2

        # draw zones
        cv2.line(frame, (LINE_X, 0), (LINE_X, frame_h), (255, 0, 0), 2)
        cv2.rectangle(frame, (LEFT_ZONE, 0), (RIGHT_ZONE, frame_h), (255, 255, 0), 2)

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

                x1, y1, x2, y2 = map(int, xyxy)
                cls_id = int(cls_id)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cls_name = names[cls_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                label = f"{cls_name} ID {track_id}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # determine zone state
                inside_zone = LEFT_ZONE <= cx <= RIGHT_ZONE

                if track_id not in object_state:
                    object_state[track_id] = {
                        "inside_zone": False,
                        "entry_time": None,
                        "direction": None
                    }

                state = object_state[track_id]

                # ENTRY
                if inside_zone and not state["inside_zone"]:
                    state["inside_zone"] = True
                    state["entry_time"] = time.time()

                # EXIT
                elif not inside_zone and state["inside_zone"]:

                    exit_time = time.time()
                    dwell_time = exit_time - state["entry_time"]

                    # determine crossing direction
                    side = get_side(cx, LINE_X)

                    if side == "right":
                        direction = "RIGHT"
                    else:
                        direction = "LEFT"

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                    log_event(timestamp, track_id, cls_name, direction, round(dwell_time, 2))

                    print(
                        f"{timestamp} | ID {track_id} | {cls_name} | {direction} | {dwell_time:.2f}s")

                    state["inside_zone"] = False
                    state["entry_time"] = None

        cv2.imshow("Line Crossing Analytics", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()