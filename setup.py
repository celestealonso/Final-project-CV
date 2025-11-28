import cv2
import numpy as np

video = "videos/video1.MOV" # REPLACE HERE TO USE OTHER EXAMPLES

# Capture a frame for calibration
cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise RuntimeError("Video can't open") 

ret, frame = cap.read()
if not ret or frame is None:
    raise RuntimeError("Can't receive frame")

# Define image and array for points
img = frame.copy()
points_image = []

# Allow user to click 4 points in the image
def click_event_calib(event, x, y, flags, param):
    global points_image, img
    if event == cv2.EVENT_LBUTTONDOWN:
        # Don't allow more than 4 points
        if len(points_image) >= 4:
            print("Only 4 points needed")
            return
        points_image.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration frame", img)

# Set window
cv2.namedWindow("Calibration frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibration frame", 800, 600)

cv2.imshow("Calibration frame", img)
cv2.setMouseCallback("Calibration frame", click_event_calib)

# Print instructions for the user
print("Click 4 points in the ground that round up the parking lot.")
print("Preferred order: bottom right, bottom left, top left, top right.")
print("Press any key to finish selecting the 4 points.")

cv2.waitKey(0)
cv2.destroyAllWindows()

# Check if the user clicked less than 4 points
points_image = np.array(points_image, dtype=np.float32)
if points_image.shape[0] != 4:
    raise RuntimeError("Less than 4 points clicked")

# Set world coordinates with arbitrary units
points_world = np.array([
    [0.0,   0.0],   # bottom right
    [100.0, 0.0],   # bottom left
    [100.0, 80.0],  # top left
    [0.0,   80.0],  # top right
], dtype=np.float32)

# Compute homography
homography, status = cv2.findHomography(points_image, points_world, method=cv2.RANSAC)

# Set resolution
pixels_per_unit = 8.0 

# Compute width and height
width_units  = points_world[:, 0].max() - points_world[:, 0].min()
height_units = points_world[:, 1].max() - points_world[:, 1].min()

bev_width  = int(width_units  * pixels_per_unit)
bev_height = int(height_units * pixels_per_unit)

x_min, y_min = points_world.min(axis=0)
x_max = points_world[:, 0].max()
y_max = points_world[:, 1].max()

# Build homography
A = np.array([
    [pixels_per_unit,  0,                  -x_min * pixels_per_unit],
    [0,               -pixels_per_unit,     y_max * pixels_per_unit],
    [0,                0,                   1]
], dtype=np.float64)

# Matrix multiplication
homography_bev = A @ homography

homography_bev_inv = np.linalg.inv(homography_bev)

# Save calibration data
np.save("homography_bev.npy", homography_bev)
np.save("bev_size.npy", np.array([bev_width, bev_height], dtype=np.int32))
# np.save("points_image.npy", points_image)
# np.save("points_world.npy", points_world)

# Create BEV
bev_calib = cv2.warpPerspective(frame, homography_bev, (bev_width, bev_height))

# Store parking slots
slots_bev = [] 

# Copy for drawing
bev_display = bev_calib.copy()

# Set up drawing tool
dragging = False
start_point = None    # (x0, y0)
current_rect = None   # (x0, y0, x1, y1)


def draw_all_slots():
    global bev_display
    bev_display = bev_calib.copy()

    # Draw saved slots (green)
    for poly in slots_bev:
        pts = poly.astype(int).reshape((-1, 1, 2))
        cv2.polylines(bev_display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw current rectangle (red)
    if current_rect is not None:
        x0, y0, x1, y1 = current_rect
        cv2.rectangle(bev_display, (x0, y0), (x1, y1), (0, 0, 255), 1)


def click_event_slots(event, x, y, flags, param):
    global dragging, start_point, current_rect, slots_bev

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_point = (x, y)
        current_rect = (x, y, x, y)
        draw_all_slots()
        cv2.imshow("BEV Slot Editor", bev_display)

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Update preview rectangle
        x0, y0 = start_point
        current_rect = (x0, y0, x, y)
        draw_all_slots()
        cv2.imshow("BEV Slot Editor", bev_display)

    elif event == cv2.EVENT_LBUTTONUP and dragging:
        dragging = False
        x0, y0 = start_point
        x1, y1 = x, y

        # Normalize coordinates so x0 < x1, y0 < y1
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        # Avoid zero-size rectangles
        if abs(x_max - x_min) > 3 and abs(y_max - y_min) > 3:
            # Create rectangle as polygon
            poly = np.array([
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
                [x_max, y_min]
            ], dtype=np.float32)

            slots_bev.append(poly)
            print(f"Rectangle slot #{len(slots_bev)}: "
                  f"({x_min}, {y_min}) to ({x_max}, {y_max})")

        current_rect = None
        draw_all_slots()
        cv2.imshow("BEV Slot Editor", bev_display)

# Set window
cv2.namedWindow("BEV Slot Editor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("BEV Slot Editor", 800, 600)
cv2.setMouseCallback("BEV Slot Editor", click_event_slots)

draw_all_slots()
cv2.imshow("BEV Slot Editor", bev_display)

# Print instructions
print("Click and drag with left mouse to define a rectangular slot.")
print("Press 'r' to reset all slots.")
print("Press ESC to cancel (no slots).")
print("Press any other key to finish and save slots.")

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == 27: # ESC
        print("Slot definition cancelled. No slots will be used.")
        slots_bev = []
        break
    elif key == ord('r'):
        slots_bev = []
        current_rect = None
        print("All slots cleared.")
        draw_all_slots()
        cv2.imshow("BEV Slot Editor", bev_display)
    else:
        break

cv2.destroyWindow("BEV Slot Editor")

# Define array for slots in the original image
slots_img = []

for poly in slots_bev:
    # Convert slots to normal image coordinates
    ones = np.ones((poly.shape[0], 1), dtype=np.float32)
    pts_bev_h = np.hstack([poly, ones])  
    pts_img_h = (homography_bev_inv @ pts_bev_h.T).T  
    pts_img = pts_img_h[:, :2] / pts_img_h[:, 2:3]
    slots_img.append(pts_img.astype(np.float32))

# Save slots
if len(slots_img) > 0:
    np.save("slots_bev.npy", np.array(slots_bev, dtype=np.float32))
    np.save("slots_img.npy", np.array(slots_img, dtype=np.float32))
    print(f"{len(slots_img)} slots saved.")
else:
    print("No slots defined.")

# Restart video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

cv2.namedWindow("Camera with slots", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera with slots", 800, 600)

print("Press ESC to quit playback.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("End of video or cannot read frame.")

    # Draw slots on original frame
    frame_display = frame.copy()

    for poly_img in slots_img:
        if poly_img.shape[0] < 2:
            continue
        pts = poly_img.astype(int).reshape((-1, 1, 2))
        cv2.polylines(
            frame_display,
            [pts],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

    # Show window
    cv2.imshow("Camera with slots", frame_display)

    if cv2.waitKey(1) & 0xFF == 27: #ESC
        break

cap.release()
cv2.destroyAllWindows()

