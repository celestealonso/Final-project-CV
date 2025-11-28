import cv2
import numpy as np
from ultralytics import YOLO

video = "videos/video1.MOV" # REPLACE HERE TO USE OTHER EXAMPLES
model_path = "yolov8n.pt"     

# Use COCO vehicle ID's (car, motorcycle, bus, truck)
vehicles = [2, 3, 5, 7]

confidence_threshold = 0.3
iou_threshold = 0.2  # IoU threshold for marking a slot as occupied

# Load data from setup
homography_bev = np.load("homography_bev.npy")
bev_size = np.load("bev_size.npy")
bev_width, bev_height = int(bev_size[0]), int(bev_size[1])

slots_bev = np.load("slots_bev.npy")
slots_img = np.load("slots_img.npy")

# Confirm float type
slots_bev = slots_bev.astype(np.float32)
slots_img = slots_img.astype(np.float32)

# Create bounding boxes
slots_img_bbox = []
for poly_img in slots_img:
    x_min = float(poly_img[:, 0].min())
    x_max = float(poly_img[:, 0].max())
    y_min = float(poly_img[:, 1].min())
    y_max = float(poly_img[:, 1].max())
    slots_img_bbox.append([x_min, y_min, x_max, y_max])

# Compute IoU between vehicle and slot
def bbox_iou(boxA, boxB):
    # Coordinates of the intersection box
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Intersection area
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    # Box area
    areaA = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))

    if areaA <= 0 or areaB <= 0:
        return 0.0 

    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


# Design 2D occupation map
bev_map_base = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
bev_map_base[:] = (40, 40, 40)


# Load model
model = YOLO(model_path)

# Open video
cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise RuntimeError("Video can't open")

# Set Window
cv2.namedWindow("Camera with occupancy", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera with occupancy", 800, 600)

cv2.namedWindow("2D Occupancy Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("2D Occupancy Map", 600, 400)

print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("End of video or cannot read frame.")

    frame_display = frame.copy()

    # Run yolo model
    results = model(frame, imgsz=640, conf=confidence_threshold, verbose=False)
    r = results[0]

    detections_xyxy = []

    # Check boxes
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            if cls_id not in vehicles:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections_xyxy.append((x1, y1, x2, y2))

    # Determine slot occupancy
    slot_states = [False] * len(slots_img_bbox)

    for (dx1, dy1, dx2, dy2) in detections_xyxy:
        det_box = [dx1, dy1, dx2, dy2]
        for i, slot_box in enumerate(slots_img_bbox):
            if slot_states[i]:
                continue 
            iou = bbox_iou(det_box, slot_box)
            if iou > iou_threshold:
                slot_states[i] = True

    # Draw detection boxes for vehicles
    for (x1, y1, x2, y2) in detections_xyxy:
        cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

    # Draw slots colored by occupancy
    for poly_img, occupied in zip(slots_img, slot_states):
        pts = poly_img.astype(int).reshape((-1, 1, 2))
        color = (0, 0, 255) if occupied else (0, 255, 0) # red for occupied, green for available
        cv2.polylines(frame_display, [pts], isClosed=True, color=color, thickness=2)

    # Label slots with numbers
    for idx, poly_img in enumerate(slots_img):
        cx = int(poly_img[:, 0].mean())
        cy = int(poly_img[:, 1].mean())
        cv2.putText(frame_display, str(idx + 1), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Build 2D occupancy map 
    bev_map = bev_map_base.copy()

    for poly_bev, occupied in zip(slots_bev, slot_states):
        pts_bev = poly_bev.astype(int).reshape((-1, 1, 2))
        color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.fillPoly(bev_map, [pts_bev], color)

    # Label slot numbers
    for idx, poly_bev in enumerate(slots_bev):
        cx = int(poly_bev[:, 0].mean())
        cy = int(poly_bev[:, 1].mean())
        cv2.putText(bev_map, str(idx + 1), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Show windows
    cv2.imshow("Camera with occupancy", frame_display)
    cv2.imshow("2D Occupancy Map", bev_map)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
