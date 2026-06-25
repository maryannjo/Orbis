import cv2
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import pyttsx3
import time

# -----------------------------
# Apple GPU (MPS) Device Setup
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# -----------------------------
# Load ViT-GPT2 Model
# -----------------------------
print("Loading ViT-GPT2 model...")

model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)

model.to(device)

processor = ViTImageProcessor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)

tokenizer = AutoTokenizer.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)

print("Model Loaded Successfully!")


# -----------------------------
# Text To Speech Setup
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 165)

def speak(text):
    engine.say(text)
    engine.runAndWait()


# -----------------------------
# Caption Generator
# -----------------------------
def generate_caption(image):

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=40,
            num_beams=2
        )

    caption = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return caption


# -----------------------------
# Camera Setup
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("\nORBIS Fast Vision Ready")
print("Press SPACE → Capture + Describe")
print("Press Q → Quit\n")

last_spoken_time = 0
cooldown = 2


# -----------------------------
# Main Loop
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # ⭐ IMPORTANT: resizing improves Mac GPU speed massively
    frame = cv2.resize(frame, (640, 480))

    cv2.imshow("ORBIS Fast Vision", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE BAR

        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        print("Analyzing scene...")

        caption = generate_caption(image)

        print("Scene:", caption)

        # Speech output
        current_time = time.time()
        if current_time - last_spoken_time > cooldown:
            speak(caption)
            last_spoken_time = current_time

    elif key == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
