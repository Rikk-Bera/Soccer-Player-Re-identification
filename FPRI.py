import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

# PATHS

MODEL_PATH = "runs/detect/train/weights/best.pt"
VIDEO_SOURCE = "D:/VS Code/practise python/15sec_input_720p.mp4"
OUTPUT_VIDEO = "output_player.mp4"

# LOAD MODEL & VIDEO

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FULL-SCREEN ROI

ROI_POINTS = [
    (0, 0),
    (w, 0),
    (w, h),
    (0, h)
]
roi_polygon = Polygon(ROI_POINTS)

# VIDEO WRITER

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 25,
    (w, h)
)

# TRACKING VARIABLES

track_last_state = {}
counted_ids = set()
cross_count = 0
# PROCESS VIDEO

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.4,
        iou=0.5
    )

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            inside = roi_polygon.contains(Point(cx, cy))

            if track_id not in track_last_state:
                track_last_state[track_id] = False

            # Count unique people
            if not track_last_state[track_id] and inside and track_id not in counted_ids:
                cross_count += 1
                counted_ids.add(track_id)

            track_last_state[track_id] = inside

            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)

    # Draw ROI (full frame)
    cv2.polylines(frame, [np.array(ROI_POINTS)], True, (255, 0, 0), 2)

    # Draw count
    cv2.putText(
        frame,
        f"People Count: {cross_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    out.write(frame)

# CLEANUP

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Processing complete. Output saved:", OUTPUT_VIDEO)
