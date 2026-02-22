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

# ----------------------------
# 1. App setup & CORS
# ----------------------------
app = FastAPI()

# This allows your Next.js website to talk to this Python AI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# 2. Path Handling (Crucial for Railway)
# ----------------------------
# This finds the absolute path of the folder this script is in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "isic_ham_b2_best.pth"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

# Force CPU mode for Railway (Free tier has no GPU)
device = torch.device("cpu")

# ----------------------------
# 3. Load Class Names
# ----------------------------
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

FULL_CLASS_NAMES = {
    "akiec": "Actinic keratosis",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion"
}

# ----------------------------
# 4. Load Model
# ----------------------------
print(f"--- Loading model from: {MODEL_PATH} ---")

model = timm.create_model(
    "efficientnet_b2",
    pretrained=False,
    num_classes=len(class_names)
)

# The 'map_location=device' is what stops the "No CUDA found" crash
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

print("--- AI Model Loaded Successfully! ---")

# ----------------------------
# 5. Image Preprocessing
# ----------------------------
IMG_SIZE = 260
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ----------------------------
# 6. Predict Endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and open image
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Transform image for AI
    x = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Get Top 3 Results
    top3_idx = probs.argsort()[-3:][::-1]

    results = []
    for i in top3_idx:
        code = class_names[int(i)]
        name = FULL_CLASS_NAMES.get(code, code)
        pct = float(probs[int(i)] * 100)

        results.append({
            "code": code,
            "name": name,
            "pct": round(pct, 2)
        })

    return {"results": results}

@app.get("/")
def health_check():
    return {"status": "AI is online and healthy"}
