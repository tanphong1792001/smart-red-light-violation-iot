import io
import random
import os

import deep_sort
from deep_sort import nn_matching
from deep_sort import Tracker
from deep_sort.tools import generate_detections as gdet
from deep_sort import Detection
from ultralytics import YOLO
from tqdm import tqdm
import time

import cv2
import easyocr
import numpy as np
import tempfile
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import numpy as np
from utils import get_intersection_area, COLORS, TrafficLightState, Box
from config import (
    ZEBRA_CROSSING,
    ZEBRA_THRESHOLD,
    VEHICLE_DET_THRESHOLD,
    LINE_THICKNESS,
    TRAFFIC_LIGHT_THRESHOLD,
)
import uuid

app = FastAPI()

# Load a pretrained YOLO11n model
coco_model = YOLO("yolo11n.pt")
traffic_light_model_med = YOLO("traffic_light_yolo_v8.pt")
traffic_light_model_small = YOLO("traffic_light_small_yolo.pt")
license_plate_model = YOLO("license_plate_detector.pt")


print("Load EasyOCR model from Internet")
reader = easyocr.Reader(["en"], gpu=False)
# Pre-defined colors for common COCO classes (as an example)
class_colors = {
    0: (0, 0, 255),  # person: red
    1: (0, 255, 0),  # bicycle: green
    2: (255, 0, 0),  # car: blue
    3: (0, 255, 255),  # motorcycle: yellow
    5: (255, 0, 255),  # bus: purple
    # Add more classes as needed
}

# Projection matrix from Image frame to BEV frame
H = np.array(
    [
        [1.11111111e-02, 3.10862447e-19, -1.11111111e01],
        [-6.34413157e-18, -6.94444444e-02, 1.00000000e02],
        [3.66675140e-19, -1.38888889e-04, 1.00000000e00],
    ],
    dtype=np.float64,
)

# speed adjustmen factor
adjust_factor = 1.5

# Load encoder
encoder = gdet.create_box_encoder("deep_sort/ckpt/mars-small128.pb", batch_size=32)

# Tracker
metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, 100)
tracker = Tracker(metric, n_init=3)


# Then use a simple get_color function
def get_color(class_id, class_names):
    if class_id in class_colors:
        return class_colors[class_id]
    else:
        # Generate a random color for unknown classes
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


@app.post("/predict_single/")
async def predict_single(
    type: str = Form(...), item: UploadFile = File(...)
) -> StreamingResponse:
    selected_idx = list(class_colors.keys())

    print(f"Received type: {type}")
    if type == "vehicle":
        selected_idx = [2, 3, 5]
        model = coco_model
    elif type == "traffic_light":
        model = traffic_light_model_med
    elif type == "license_plate":
        model = license_plate_model

    # Read the image file
    image_bytes = await item.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_with_boxes = image.copy()
    # Run inference on the image
    results = model(image)

    # Process the results
    for result in results:
        boxes = result.boxes  # Boxes object for bounding boxes outputs

        # Get the boxes, labels, and confidence scores
        for box, cls, conf in zip(
            boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            label = result.names[int(cls)]  # Get class name
            confidence = float(conf)  # Get confidence score
            class_id = int(cls)

            if class_id not in selected_idx:
                continue

            # Get color for this class
            color = get_color(class_id, result.names)

            # Draw bounding box
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)

            # Create label text with class name and confidence
            label_text = f"{label}: {confidence:.2f}"

            # Calculate text size and position
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
            )
            cv2.rectangle(
                image_with_boxes, (x1, y1 - 20), (x1 + text_width, y1), color, -1
            )

            # Draw text
            cv2.putText(
                image_with_boxes,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    # Convert the image to bytes
    _, encoded_image = cv2.imencode(".jpg", image_with_boxes)
    io_buf = io.BytesIO(encoded_image.tobytes())
    io_buf.seek(0)

    # Return the image as a StreamingResponse
    return StreamingResponse(io_buf, media_type="image/jpeg")


@app.post("/ocr_run/")
async def ocr_run(item: UploadFile = File(...)) -> JSONResponse:
    item = await item.read()
    text = reader.readtext(item, detail=0)
    print(text)
    return JSONResponse(content={"result": text})


@app.post("/tracking")
async def tracking(request: Request):
    video_bytes = await request.body()  # Get the raw payload

    # Save the video to a file (e.g., with a unique name)
    filename = f"video.mp4"
    with open(filename, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"VP80")  # Use 'XVID' or 'avc1' for .avi/.mp4
    out = cv2.VideoWriter("output.webm", fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="[Tracking] Processing frames", unit="frame")

    def convert_pixel_to_bev(x, y, H):
        """
        Convert pixel coordinates to Bird's Eye View (BEV) coordinates.
        """
        pixel = np.array([x, y, 1], dtype=np.float32)
        bev = np.dot(H, pixel)
        return int(bev[0] / bev[2]), int(bev[1] / bev[2])

    tracked_pose_dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()

        results = coco_model(frame, verbose=False)[0]  # Only 1 frame

        # Extract detections
        boxes = []
        confidences = []
        for box, conf, cls in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
        ):
            if conf < 0.4:
                continue
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            boxes.append([int(x1), int(y1), int(w), int(h)])
            confidences.append(float(conf))

        # Encode features
        features = encoder(frame, boxes)

        # Create detections
        detections = [
            Detection(bbox, conf, feat)
            for bbox, conf, feat in zip(boxes, confidences, features)
        ]

        # Non-Maximum Suppression
        boxes_np = np.array([d.tlwh for d in detections])
        scores_np = np.array([d.confidence for d in detections])

        # Update tracker
        tracker.predict()
        tracker.update(detections)

        # Draw tracks
        current_vel = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h = track.to_tlwh()
            x_bev, y_bev = convert_pixel_to_bev(x, y, H)
            timestamp = time.time()  # Get the current timestamp
            track_id = track.track_id

            if tracked_pose_dict.get(track_id) is None:
                vel_kmh = 0
                tracked_pose_dict[track_id] = [x_bev, y_bev, timestamp, vel_kmh]
            else:
                vel_x = x_bev - tracked_pose_dict[track_id][0]
                vel_y = y_bev - tracked_pose_dict[track_id][1]
                previous_vel = tracked_pose_dict[track_id][3]
                duration = timestamp - tracked_pose_dict[track_id][2]

                tracked_pose_dict[track_id][0] = x_bev
                tracked_pose_dict[track_id][1] = y_bev
                tracked_pose_dict[track_id][2] = timestamp
                vel_ms = np.sqrt(vel_x**2 + vel_y**2) / duration
                vel_kmh = vel_ms * 3.6 * adjust_factor

                if vel_kmh == 0:
                    vel_kmh = tracked_pose_dict[track_id][3]
                else:
                    tracked_pose_dict[track_id][3] = vel_kmh

            cv2.rectangle(
                frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"ID {track_id}",
                (int(x), int(y - 10)),
                0,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"V: {vel_kmh:.2f} km/h",
                (int(x), int(y + h + 10)),
                0,
                0.6,
                (255, 255, 255),
                2,
            )

        end = time.perf_counter()
        fps = 1 / (end - start)
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        out.write(frame)  # Write the frame with tracking to the output video
        pbar.update(1)  # Update progress bar
    cap.release()
    out.release()

    # Clean up temporary files
    if os.path.exists(filename):
        os.remove(filename)

    return StreamingResponse(
        open("output.webm", "rb"),
        media_type="video/webm",
        headers={"Content-Disposition": 'inline; filename="output.webm"'},
    )


@app.post("/red_light_violation")
async def red_light_violation(request: Request):
    video_bytes = await request.body()  # Get the raw payload

    # Save the video to a file (e.g., with a unique name)
    filename = f"video.mp4"

    with open(filename, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"VP80")  # Use 'XVID' or 'avc1' for .avi/.mp4
    out = cv2.VideoWriter("output.webm", fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    pbar = tqdm(
        total=total_frames, desc="[RED LIGHT VIOLATION] Processing frames", unit="frame"
    )

    violation_db = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results_vehicle_det = coco_model(frame, verbose=False)[0]  # Only 1 frame
        result_traffic_light = traffic_light_model_med(frame, verbose=False)[
            0
        ]  # Only 1 frame

        # Check if traffic light is red
        is_red_light = False
        if (
            result_traffic_light.boxes is not None
            and len(result_traffic_light.boxes) > 0
        ):
            traffic_light_boxes = result_traffic_light.boxes.xyxy.cpu().numpy()
            traffic_light_classes = result_traffic_light.boxes.cls.cpu().numpy()
            traffic_light_confidences = result_traffic_light.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(
                traffic_light_boxes, traffic_light_classes, traffic_light_confidences
            ):
                if conf < TRAFFIC_LIGHT_THRESHOLD:
                    continue

                if cls == TrafficLightState.RED.code:  # Red light detected
                    is_red_light = True

        # Extract detections
        boxes = []
        confidences = []
        for box, conf, cls in zip(
            results_vehicle_det.boxes.xyxy.cpu().numpy(),
            results_vehicle_det.boxes.conf.cpu().numpy(),
            results_vehicle_det.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            if int(cls) == 9:
                # draw traffic light box
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    COLORS.BLUE.value,
                    LINE_THICKNESS,
                )

                
                cv2.putText(
                    frame,
                    TrafficLightState.get_label_by_code(traffic_light_classes[0]),
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLORS.WHITE.value,
                    LINE_THICKNESS,
                )

            if conf < VEHICLE_DET_THRESHOLD or cls not in class_colors:
                continue

            w, h = x2 - x1, y2 - y1
            boxes.append([int(x1), int(y1), int(w), int(h)])
            confidences.append(float(conf))

        # Encode features
        features = encoder(frame, boxes)

        # Create detections
        detections = [
            Detection(bbox, conf, feat)
            for bbox, conf, feat in zip(boxes, confidences, features)
        ]

        # Non-Maximum Suppression
        boxes_np = np.array([d.tlwh for d in detections])
        scores_np = np.array([d.confidence for d in detections])

        # Update tracker
        tracker.predict()
        tracker.update(detections)

        # Draw tracks
        current_vel = 0
        is_line_oversteped = False

        area = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h = track.to_tlwh()
            object_box = Box(top_left=(x, y), bottom_right=(x + w, y + h))

            object_box.top_left = [
                object_box.top_left[0],
                object_box.top_left[1] + int(h / 3),
            ]

            object_box.bottom_right = [
                object_box.bottom_right[0],
                object_box.bottom_right[1] + int(h / 3),
            ]

            area = get_intersection_area(object_box, ZEBRA_CROSSING)
            if area > ZEBRA_THRESHOLD:
                is_line_oversteped = True
                violation_db.add(track.track_id)

            track_id = track.track_id

            # draw bounding box with v green
            cv2.rectangle(
                frame,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                COLORS.GREEN.value,
                LINE_THICKNESS,
            )
            cv2.putText(
                frame,
                f"ID {track_id}",
                (int(x), int(y - 10)),
                0,
                0.6,
                COLORS.WHITE.value,
                LINE_THICKNESS,
            )

        if is_line_oversteped and is_red_light:
            string_violation = ""
            for item in violation_db:
                string_violation += f"ID-{item}, "
            cv2.putText(
                frame,
                f"Violation Detected: {string_violation}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                COLORS.RED.value,
                2,
            )
            zebra_color = COLORS.RED.value
        else:
            zebra_color = COLORS.GREEN.value

        cv2.rectangle(
            frame,
            ZEBRA_CROSSING.top_left,
            ZEBRA_CROSSING.bottom_right,
            zebra_color,
            LINE_THICKNESS,
        )
        # draw area
        out.write(frame)
        pbar.update(1)

    out.release()
    cap.release()

    # Clean up temporary files
    if os.path.exists(filename):
        os.remove(filename)
    return StreamingResponse(
        open("output.webm", "rb"),
        media_type="video/webm",
        headers={"Content-Disposition": 'inline; filename="output.webm"'},
    )
