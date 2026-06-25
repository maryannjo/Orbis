import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


import cv2
import pytesseract
import pandas as pd
import pyttsx3
import re
import numpy as np


# ---------- LOAD IMAGE ----------
img = cv2.imread(r"/Users/maryannjoseph/Downloads/bus.jpg.jpg")

if img is None:
    print("Image not loaded")
    exit()

# ---------- CROP BUS LED DISPLAY ----------
y1, y2 = 30, 170
x1, x2 = 300, 1200
bus_region = img[y1:y2, x1:x2]

# ---------- PREPROCESS ----------
gray = cv2.cvtColor(bus_region, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

thresh = cv2.bitwise_not(thresh)

# ---------- OCR ----------
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ocr_text = pytesseract.image_to_string(thresh, config=custom_config)
ocr_text = ocr_text.upper().strip()

print("OCR Text:", ocr_text)

# ---------- SMART BUS NUMBER EXTRACTION ----------
# This will extract 104C even from '104CXSUDUVAN'
bus_match = re.search(r'\d{1,3}[A-Z]', ocr_text)

if not bus_match:
    print("No bus number detected")
    exit()

bus_no = bus_match.group()
print("Detected Bus Number:", bus_no)

# ---------- LOAD DATASET ----------
df = pd.read_csv("/Users/maryannjoseph/Downloads/bus_routes.csv")
df.columns = df.columns.str.strip().str.lower()

# ---------- SAFE MATCH ----------
result = df[df["busnumber"].str.upper().str.contains(bus_no)]

if result.empty:
    print("Bus not found in dataset")
    exit()

route = result.iloc[0]["route"]

# ---------- FINAL OUTPUT ----------
print("\nBus Number:", bus_no)
print("Route:", route)

# ---------- VOICE OUTPUT ----------
engine = pyttsx3.init()
engine.say(f"Bus number {bus_no}. Route {route}")
engine.runAndWait()