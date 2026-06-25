import os
from transformers import BlipProcessor, BlipForQuestionAnswering

# 1. Define the variables FIRST
MODEL_ID = "Salesforce/blip-vqa-base"
SAVE_PATH = "./models/blip_vqa"

# 2. Increase timeout for slow connections
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300" 

print("Starting download... This may take a few minutes.")

# 3. Use the variables to download from the internet
# Note: We do NOT use local_files_only=True here because we are currently downloading THEM.
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForQuestionAnswering.from_pretrained(MODEL_ID)

# 4. Save them to your folder
processor.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"✅ Success! Models saved to {SAVE_PATH}")