import os
import io
import json
import torch
import timm
import torchvision.transforms as T
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()

# Allows your Next.js website to communicate with this AI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Path & Device Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "isic_ham_b2_best.pth")
device = torch.device("cpu") # Railway free tier uses CPU only

# --- Load Data & Model ---
with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)

FULL_CLASS_NAMES = {
    "akiec": "Actinic keratosis", "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion", "df": "Dermatofibroma",
    "mel": "Melanoma", "nv": "Melanocytic nevus", "vasc": "Vascular lesion"
}

model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=len(class_names))
state = torch.load(MODEL_PATH, map_location=device) # Load to CPU
model.load_state_dict(state)
model.eval()

# --- Image Setup ---
transform = T.Compose([
    T.Resize((260, 260)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    
    top3_idx = probs.argsort()[-3:][::-1]
    results = [{"code": class_names[int(i)], 
                "name": FULL_CLASS_NAMES.get(class_names[int(i)], class_names[int(i)]), 
                "pct": round(float(probs[int(i)] * 100), 2)} for i in top3_idx]
    return {"results": results}

@app.get("/")
def health(): return {"status": "online"}
