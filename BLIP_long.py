import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Loading ORBIS Scene Analysis Model...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl"
).to(device)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Camera ready. Press SPACE to analyze scene. Press Q to quit.")

def generate_scene_breakdown(image):
    prompt = """
You are an AI assistant for smart glasses helping a visually impaired user.

Analyze the scene and respond in this structured format:

1. Main Scene: What is happening overall
2. People: Number of people, appearance, clothing, actions
3. Objects: Important objects and their positions
4. Environment: Indoor/outdoor, lighting, weather, setting
5. Movement: Any motion or activity happening
6. Potential Hazards: Anything the user should be careful about
7. Short Summary: 1 concise sentence

Be clear, specific, and practical.
"""

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,   # deterministic → better structured output
            temperature=0.7
        )

    return processor.decode(output[0], skip_special_tokens=True)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("ORBIS Camera", frame)
    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        print("\n🔍 Analyzing Scene...\n")
        description = generate_scene_breakdown(image)

        print(description)
        print("\n" + "="*50 + "\n")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()