import gradio as gr
import requests
import io
import numpy as np
from PIL import Image

API_URL = "http://localhost:8000/predict"

# --- Face detection (MediaPipe) ---
def detect_face_and_suggest_crop(pil_img):
    if pil_img is None:
        return None, "Aucune image."

    import cv2
    import numpy as np
    from PIL import Image

    img = pil_img.convert("RGB")
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return img, "Aucun visage détecté. Recadrez manuellement."

    # Prendre le plus grand visage
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Marge pour inclure menton / front
    margin = 0.25
    mx = int(w * margin)
    my = int(h * margin)

    W, H = img.size
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)

    crop = img.crop((x1, y1, x2, y2))
    return crop, "Visage détecté ✅ Ajustez le recadrage si besoin."


def _extract_pil_from_editor(editor_value):
    """
    Gradio ImageEditor renvoie souvent un dict.
    On essaye de récupérer l'image finale (composite), sinon background.
    """
    if editor_value is None:
        return None

    # Si Gradio renvoie déjà une PIL Image
    if isinstance(editor_value, Image.Image):
        return editor_value

    # Si c'est un dict (cas courant)
    if isinstance(editor_value, dict):
        # Le plus probable : "composite" (image finale)
        if editor_value.get("composite") is not None:
            return editor_value["composite"]
        # Sinon "background"
        if editor_value.get("background") is not None:
            return editor_value["background"]

    return None



def preprocess_and_send(editor_value):
    pil_crop = _extract_pil_from_editor(editor_value)
    if pil_crop is None:
        return {"Veuillez recadrer / fournir un visage": 1.0}

    img = pil_crop.convert("RGB")
    img = img.resize((96, 96), Image.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    payload = buf.getvalue()

    files = {"file": ("face.jpg", payload, "image/jpeg")}

    try:
        r = requests.post(API_URL, files=files, timeout=10)
        r.raise_for_status()
        return r.json()["probs"]
    except Exception:
        return {"Erreur de connexion à l'API": 1.0}

def debug(editor_value):
    return str(type(editor_value)), str(editor_value.keys()) if isinstance(editor_value, dict) else "not a dict"


with gr.Blocks(title="Détection d'Émotions (ResNet50)") as demo:
    gr.Markdown(
        "# Détection d'Émotions (ResNet50)\n"
        "1) Upload → 2) détection visage + suggestion de crop → 3) ajustez → 4) prédire"
    )

    with gr.Row():
        inp = gr.Image(type="pil", label="1) Uploader une photo")
        editor = gr.ImageEditor(
            type="pil",
            label="2) Visage recadré (vous pouvez ajuster ici)",
        )

    status = gr.Markdown("")

    with gr.Row():
        btn_detect = gr.Button("Détecter visage / Proposer un recadrage")
        btn_predict = gr.Button("Prédire")

    out = gr.Label(num_top_classes=3, label="Prédictions")

    # Quand on clique "Détecter", on remplit l'éditeur avec le crop suggéré
    btn_detect.click(
        fn=detect_face_and_suggest_crop,
        inputs=inp,
        outputs=[editor, status],
    )

    # Option pratique : auto-détection à l'upload
    inp.change(
        fn=detect_face_and_suggest_crop,
        inputs=inp,
        outputs=[editor, status],
    )

    # Quand on clique "Prédire", on envoie l'image issue de l'éditeur
    btn_predict.click(
        fn=preprocess_and_send,
        inputs=editor,
        outputs=out,
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
