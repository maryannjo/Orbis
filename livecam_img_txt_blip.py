import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Loading AI model...")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Camera ready. Press SPACE to capture. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("ORBIS Camera", frame)
    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs)

        caption = processor.decode(output[0], skip_special_tokens=True)
        print("Description:", caption)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
