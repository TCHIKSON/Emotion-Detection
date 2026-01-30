import json
import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications import resnet

preprocess_input = resnet.preprocess_input

app = FastAPI()

# ---------------------------------------------------------
# Configuration CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Chemins absolus
# ---------------------------------------------------------
# .../Emotion-Detection/font/api
API_DIR = os.path.dirname(os.path.abspath(__file__))

# .../Emotion-Detection/font
FONT_DIR = os.path.dirname(API_DIR)

# .../Emotion-Detection  <-- RACINE DU PROJET
PROJECT_DIR = os.path.dirname(FONT_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "best_resnet50_fer_finetuned.keras")
CLASSES_PATH = os.path.join(PROJECT_DIR, "class_names.json")

IMG_SIZE = (96, 96)

# ---------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"⚠️ Modèle introuvable à l'emplacement : {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


with open(CLASSES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


# ---------------------------------------------------------
# Préparation de l'image
# ---------------------------------------------------------
def prepare_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    x = np.array(img).astype(np.float32)
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    x = preprocess_input(x)        # Normalisation ResNet

    return x
# ---------------------------------------------------------
# Endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    b = await file.read()
    x = prepare_image(b)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))

    return {
        "class": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }
