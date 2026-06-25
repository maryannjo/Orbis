import torch
import cv2
import os
import numpy as np
import pyttsx3
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# --- 1. M4 ACCELERATION --- 
device = torch.device("mps")
print(f" M4 Neural Engine Detected. Optimization: ACTIVE.")

# Initialize Voice Engine (Offline)
engine = pyttsx3.init()
engine.setProperty('rate', 180) # Speed of speech

def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- 2. LOCAL MODEL LOADING ---
BLIP_PATH = "./models/blip_vqa"
MIDAS_CACHE = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")

print("Waking up the M4 Brain...")
processor = BlipProcessor.from_pretrained(BLIP_PATH, local_files_only=True)
vqa_model = BlipForQuestionAnswering.from_pretrained(
    BLIP_PATH, 
    local_files_only=True,
    torch_dtype=torch.float16, # M4 loves Float16 math
    low_cpu_mem_usage=True
).to(device)

midas = torch.hub.load(MIDAS_CACHE, "MiDaS_small", source="local", trust_repo=True).to(device).eval()
midas_transforms = torch.hub.load(MIDAS_CACHE, "transforms", source="local")
transform = midas_transforms.small_transform

def start_orbis_m4():
    cap = cv2.VideoCapture(0)
    # M4 can handle higher resolution input
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    DANGER_THRESHOLD = 700 
    previous_alert = ""

    print("--- ORBIS M4 LIVE: VOICE READY ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # M4 Customization: Process every 2nd frame (2x faster than previous version)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 2 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=frame.shape[:2],
                    mode="bicubic", align_corners=False
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            h, w = depth_map.shape
            walking_zone = depth_map[int(h*0.7):, int(w*0.3):int(w*0.7)]
            proximity = np.max(walking_zone)

            if proximity > DANGER_THRESHOLD: 
                raw_image = Image.fromarray(img_rgb)
                inputs = processor(raw_image, "What is it?", return_tensors="pt").to(device)
                
                with torch.no_grad():
                    out = vqa_model.generate(**inputs, max_new_tokens=8)
                    obstacle = processor.decode(out[0], skip_special_tokens=True)
                
                current_alert = f"Stop. {obstacle.upper()} ahead."
            else:
                current_alert = "Path Clear"

            # --- VOICE & TERMINAL LOGIC ---
            if current_alert != previous_alert:
                print(f"[ORBIS]: {current_alert}")
                # Use the M4 to speak the alert
                if "Stop" in current_alert:
                    speak(current_alert)
                previous_alert = current_alert
            
            # Keep M4 memory clean
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        cv2.imshow('Orbis M4 Prototype', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_orbis_m4()