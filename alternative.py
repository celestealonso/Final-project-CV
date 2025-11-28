import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, ViTForImageClassification

video = "videos/video1.MOV"  # REPLACE HERE TO USE OTHER EXAMPLES

# Set up ViT
vit_model_name = "vit-pklot"   
slot_crop_size = 224           

# Set ID label for occupied
occupied_label_id = 1          

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data from setup
homography_bev = np.load("homography_bev.npy")
bev_size = np.load("bev_size.npy")
bev_width, bev_height = int(bev_size[0]), int(bev_size[1])

slots_bev = np.load("slots_bev.npy")
slots_img = np.load("slots_img.npy")

# Confirm float type
slots_bev = slots_bev.astype(np.float32)
slots_img = slots_img.astype(np.float32)

# Design 2D occupation map
bev_map_base = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
bev_map_base[:] = (40, 40, 40)

# Load trained ViT
vit_processor = AutoImageProcessor.from_pretrained(vit_model_name)
vit_model = ViTForImageClassification.from_pretrained(vit_model_name).to(device)
vit_model.eval()

def warp_slot_patch(frame, poly_img, out_size=224):
    src = poly_img.astype(np.float32)
    
    # Transform the tilted slot into a square
    dst = np.array([
        [out_size - 1, out_size - 1],  
        [0,            out_size - 1],  
        [0,            0],             
        [out_size - 1, 0],            
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (out_size, out_size))
    return warped

# Open video
cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise RuntimeError("Video can't open")

# Set windows
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

    # Build patches for the ViT
    slot_patches = []
    valid_indices = [] 

    # Loop through the slots and warp them
    for idx, poly_img in enumerate(slots_img):
        patch = warp_slot_patch(frame, poly_img, out_size=slot_crop_size)
        if patch is None or patch.size == 0:
            continue
        slot_patches.append(patch)
        valid_indices.append(idx)

    # Initialize all slots as free
    slot_states = [False] * len(slots_img)

    # Run ViT on all patches
    if slot_patches:
        with torch.no_grad():
            # Process into tensor and run ViT
            inputs = vit_processor(images=slot_patches, return_tensors="pt").to(device)
            outputs = vit_model(**inputs)
            logits = outputs.logits 
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        # Input predictions into slots
        for k, slot_idx in enumerate(valid_indices):
            pred_label = int(preds[k])
            occupied = (pred_label == occupied_label_id)
            slot_states[slot_idx] = occupied

    # Draw slots colored by occupancy
    for poly_img, occupied in zip(slots_img, slot_states):
        pts = poly_img.astype(int).reshape((-1, 1, 2))
        color = (0, 0, 255) if occupied else (0, 255, 0)  # red for occupied, green for available
        cv2.polylines(frame_display, [pts], isClosed=True, color=color, thickness=2)

    # Label slots with numbers
    for idx, poly_img in enumerate(slots_img):
        cx = int(poly_img[:, 0].mean())
        cy = int(poly_img[:, 1].mean())
        cv2.putText(
            frame_display,
            str(idx + 1),
            (cx - 5, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

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
        cv2.putText(
            bev_map,
            str(idx + 1),
            (cx - 5, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Show windows
    cv2.imshow("Camera with occupancy", frame_display)
    cv2.imshow("2D Occupancy Map", bev_map)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
