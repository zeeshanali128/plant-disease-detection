import os
import sys
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import *


_model = None
_class_names = None


def load_assets():
    global _model, _class_names
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first.")
        _model = load_model(MODEL_PATH)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
    if _class_names is None:
        with open(CLASS_JSON, 'r') as f:
            _class_names = json.load(f)
        print(f"[INFO] {len(_class_names)} classes loaded.")
    return _model, _class_names


def predict_image(img_path, top_k=3):
    """
    Predicts disease for a single image file.

    Returns:
        dict with keys: class_name, confidence, top_k_predictions
    """
    model, class_names = load_assets()

    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_predictions = [
        {
            "rank":       i + 1,
            "class_name": class_names[idx],
            "confidence": round(float(probs[idx]) * 100, 2)
        }
        for i, idx in enumerate(top_indices)
    ]

    best = top_predictions[0]

    raw = best["class_name"]
    if "___" in raw:
        parts = raw.split("___")
        crop    = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip()
    elif "__" in raw:
        parts = raw.split("__")
        crop    = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip()
    else:
        tokens  = raw.replace("_", " ").split()
        crop    = tokens[0]
        disease = " ".join(tokens[1:]) if len(tokens) > 1 else "Unknown"

    parts = f"{crop} — {disease}"

    return {
        "class_name":      best["class_name"],
        "display_name":    parts,
        "crop":            crop.strip(),
        "disease":         disease.strip() if disease else "Healthy",
        "confidence":      best["confidence"],
        "is_healthy":      "healthy" in best["class_name"].lower(),
        "top_k":           top_predictions
    }


if __name__ == "__main__":
    import glob

    test_imgs = glob.glob(os.path.join(TEST_DIR, '**', '*.jpg'), recursive=True)
    if test_imgs:
        result = predict_image(test_imgs[0])
        print("\n=== PREDICTION RESULT ===")
        print(f"  Crop    : {result['crop']}")
        print(f"  Disease : {result['disease']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Healthy : {result['is_healthy']}")
        print("\nTop-3 predictions:")
        for p in result['top_k']:
            print(f"  {p['rank']}. {p['class_name']:<45} {p['confidence']:.2f}%")
    else:
        print("No test images found.")