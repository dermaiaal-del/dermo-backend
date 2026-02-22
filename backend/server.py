from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as T

# ----------------------------
# App setup
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Next.js + phone
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load class names
# ----------------------------
with open("class_names.json", "r") as f:
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

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load model
# ----------------------------
model = timm.create_model(
    "efficientnet_b2",
    pretrained=False,
    num_classes=len(class_names)
)

state = torch.load("isic_ham_b2_best.pth", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# ----------------------------
# Preprocess
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
# Predict endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Top 3
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