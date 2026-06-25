import cv2
import torch
import numpy as np
from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

import easyocr
from ultralytics import YOLO
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

print("Loading BLIP...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

print("Loading OCR...")

ocr_reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("Loading YOLO...")

yolo_model = YOLO("yolo11n.pt")

if device == "mps":
    yolo_model.to("mps")

print("All models loaded!")


def enhance_image(frame):

    enhanced = cv2.convertScaleAbs(
        frame,
        alpha=1.2,
        beta=20
    )

    denoised = cv2.fastNlMeansDenoisingColored(
        enhanced,
        None,
        10,
        10,
        7,
        21
    )

    return denoised

def read_text(frame):

    results = ocr_reader.readtext(frame)

    texts = []

    for result in results:
        texts.append(result[1])

    return texts

def detect_objects(frame):

    results = yolo_model.predict(
        source=frame,
        verbose=False
    )

    objects = set()

    for r in results:

        for box in r.boxes:

            cls = int(box.cls)

            name = yolo_model.names[cls]

            objects.add(name)

    return list(objects)


def generate_caption(frame):

    image = Image.fromarray(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
    )

    inputs = processor(
        image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        output = blip_model.generate(
            **inputs,
            max_new_tokens=40
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption

def overlay_edges(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edges_bgr = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )

    combined = cv2.addWeighted(
        frame,
        0.85,
        edges_bgr,
        0.15,
        0
    )

    return combined


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("\n==============================")
print("ORBIS READY")
print("SPACE -> Analyze Frame")
print("Q -> Quit")
print("==============================\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Full color camera feed
    display_frame = overlay_edges(frame)

    cv2.imshow(
        "ORBIS Camera",
        display_frame
    )

    key = cv2.waitKey(1)


    if key == 32:

        print("\nAnalyzing...\n")

        processed = enhance_image(frame)

        caption = generate_caption(processed)

        objects = detect_objects(processed)

        text_found = read_text(processed)

        print("====================================")
        print("SCENE DESCRIPTION")
        print("====================================")
        print(caption)

        print("\nDETECTED OBJECTS")

        if len(objects):
            for obj in objects:
                print("-", obj)
        else:
            print("No objects detected")

        print("\nDETECTED TEXT")

        if len(text_found):
            for text in text_found:
                print("-", text)
        else:
            print("No text detected")

        print("====================================\n")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()