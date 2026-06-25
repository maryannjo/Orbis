from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# =======================
# Initialize FastAPI
# =======================
app = FastAPI()

# =======================
# Device configuration (Mac GPU - MPS)
# =======================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =======================
# Load MiDaS for depth estimation
# =======================
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

# MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# =======================
# Load YOLOv8 for object detection
# =======================
yolo_model = YOLO("yolov8n.pt")  # Nano model for speed

# =======================
# Depth smoothing variables
# =======================
prev_depth = None
alpha = 0.9

# =======================
# Homepage route
# =======================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>ORBIS Depth + YOLO</title>
        </head>
        <body>
            <h1>ORBIS Depth + YOLO</h1>
            <img src="/video" width="640" height="480">
        </body>
    </html>
    """

# =======================
# Video stream generator
# =======================
def generate_frames():
    global prev_depth

    cap = cv2.VideoCapture(0, cv2.CAP_FOUNDATION if hasattr(cv2, 'CAP_FOUNDATION') else 0)
    if not cap.isOpened():
        print("❌ Camera not working")
        return

    print("✅ Depth + YOLO stream started")

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Frame not captured")
            break

        try:
            # -----------------------
            # MiDaS depth estimation
            # -----------------------
            img = cv2.resize(frame, (256, 256))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            input_batch = transform(img_rgb).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            # Smooth depth
            if prev_depth is None:
                prev_depth = depth_map
            else:
                depth_map = alpha * prev_depth + (1 - alpha) * depth_map
                prev_depth = depth_map

            # Normalize depth
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

            # Center depth value
            h, w = depth_map.shape
            cx, cy = w // 2, h // 2
            center_depth = depth_map[cy, cx]

            if center_depth > 0.75:
                dist_label = "Very Close"
            elif center_depth > 0.55:
                dist_label = "Close"
            elif center_depth > 0.35:
                dist_label = "Medium"
            else:
                dist_label = "Far"

            # Depth visualization
            depth_uint8 = (depth_map * 255).astype("uint8")
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
            depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
            output = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)

            # Draw center point
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
            cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)

            # Display depth text
            cv2.putText(
                output,
                f"Depth: {dist_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                output,
                f"Value: {center_depth:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # -----------------------
            # YOLO object detection
            # -----------------------
            results = yolo_model(frame)  # use original frame
            res = results[0]

            if hasattr(res, 'boxes') and res.boxes is not None:
                for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    conf = float(conf.cpu().numpy())
                    cls = int(cls.cpu().numpy())
                    label = f"{res.names[cls]} {conf:.2f}"

                    # Draw boxes and labels (green for standard YOLO)
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # -----------------------
            # Encode frame and yield
            # -----------------------
            success, buffer = cv2.imencode('.jpg', output)
            if not success:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

        except Exception as e:
            print("🔥 ERROR:", e)
            continue

    cap.release()

# =======================
# Video stream route
# =======================
@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )