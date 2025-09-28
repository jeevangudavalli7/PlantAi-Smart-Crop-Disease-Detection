# backend/app.py
import os
import io
import logging
import json
import re
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Try tensorflow import
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    raise RuntimeError("TensorFlow import failed. Install with `pip install tensorflow`. "
                       f"Original error: {e}")

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
INDEX_HTML = os.path.join(BASE_DIR, "index.html")      # frontend here
DATASET_DIR = os.path.join(BASE_DIR, "PlantVillage", "train")
CLASSES_TXT = os.path.join(BASE_DIR, "classes.txt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
TRANSLATIONS_DIR = os.path.join(BASE_DIR, "translations")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
DEBUG = True
# ---------------------------------------

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSLATIONS_DIR, exist_ok=True)
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger("plant-detection")

app = Flask(__name__, static_folder=None)
CORS(app)  # enable CORS for local development; adjust for production
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

MODEL = None
CLASS_NAMES = []
INPUT_SHAPE = (224, 224, 3)
_TRANS_CACHE = {}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_class_names():
    if os.path.exists(CLASSES_TXT):
        try:
            with open(CLASSES_TXT, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
                if classes:
                    logger.info(f"Loaded {len(classes)} classes from classes.txt")
                    return classes
        except Exception:
            logger.exception("reading classes.txt failed")
    if os.path.isdir(DATASET_DIR):
        try:
            classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
            if classes:
                logger.info(f"Inferred {len(classes)} classes from dataset folder")
                return classes
        except Exception:
            logger.exception("listing dataset folder failed")
    logger.warning("No class names found.")
    return []

def get_model_input_size(model):
    try:
        shape = None
        if hasattr(model, "input_shape") and model.input_shape is not None:
            shape = model.input_shape
        elif hasattr(model, "inputs") and model.inputs:
            shp = model.inputs[0].shape
            shape = tuple([int(x) if x is not None else None for x in shp.as_list()])
        logger.debug(f"Detected raw input shape: {shape}")
        if shape:
            if len(shape) == 4:
                _, a, b, c = shape
                return (int(a), int(b), int(c))
            if len(shape) == 3:
                return (int(shape[0]), int(shape[1]), int(shape[2]))
    except Exception:
        logger.exception("Failed to parse model input shape")
    return (224, 224, 3)

def preprocess_pil_image(pil_img, target_size):
    h, w, ch = target_size
    pil_img = pil_img.resize((w, h))
    arr = np.asarray(pil_img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * ch, axis=-1)
    if arr.shape[-1] != ch:
        if arr.shape[-1] == 1 and ch == 3:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        else:
            if arr.shape[-1] > ch:
                arr = arr[..., :ch]
            else:
                last = np.expand_dims(arr[..., -1], -1)
                repeats = ch - arr.shape[-1]
                arr = np.concatenate([arr] + [last] * repeats, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

def decode_image_file(file_stream):
    try:
        img = Image.open(file_stream).convert("RGB")
        return img
    except Exception:
        logger.exception("Failed to decode image")
        raise

def safe_load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    logger.info(f"Loading model from {path}")
    try:
        m = load_model(path, compile=False)
        logger.info("Model loaded.")
        return m
    except Exception:
        logger.exception("Model load failed")
        raise

def simplify_class_key(name: str) -> str:
    """
    Produce a simplified short form of the predicted class name to ease matching in frontend translations.
    Examples:
      'Tomato___Late_blight' -> 'Late_blight'
      'Potato_Early_blight' -> 'Early_blight'
      'Healthy' -> 'Healthy'
    """
    if not name:
        return name
    # first try splitting on triple underscores commonly used in PlantVillage variants
    parts = re.split(r'[_]{2,}|__+|___+|\s+|-|/', name)
    # if there are recognizable separators and last part is meaningful, return last part
    if len(parts) > 1:
        candidate = parts[-1]
        candidate = candidate.strip()
        if candidate:
            return candidate
    # fallback: try to take the last token of snake case or CamelCase
    if '_' in name:
        return name.split('_')[-1]
    return name

def load_translations(lang="en"):
    if not lang:
        lang = "en"
    if lang in _TRANS_CACHE:
        return _TRANS_CACHE[lang]
    fn = os.path.join(TRANSLATIONS_DIR, f"{lang}.json")
    if not os.path.exists(fn):
        logger.debug(f"Translation file not found for {lang}: {fn}")
        _TRANS_CACHE[lang] = {}
        return {}
    try:
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
            _TRANS_CACHE[lang] = data
            logger.info(f"Loaded translations for {lang}")
            return data
    except Exception:
        logger.exception("Failed to load translation file")
        _TRANS_CACHE[lang] = {}
        return {}

def build_generic_disease_info(class_name, lang="en"):
    translations = load_translations(lang) or {}
    classes_trans = translations.get("classes", {})
    generic = translations.get("generic", {})

    cname = str(class_name)
    cname_lower = cname.lower()
    is_healthy = ("healthy" in cname_lower) or ("normal" in cname_lower)

    # localized_name: try multiple lookup strategies
    localized_name = None
    description = ""
    recommendations = []

    if cname in classes_trans:
        ctrans = classes_trans[cname]
        localized_name = ctrans.get("name", cname)
        description = ctrans.get("description", "")
        recommendations = ctrans.get("recommendations", [])
    else:
        for k, v in classes_trans.items():
            if k.lower() == cname_lower:
                localized_name = v.get("name", cname)
                description = v.get("description", "")
                recommendations = v.get("recommendations", [])
                break

    if not localized_name:
        if is_healthy:
            g = generic.get("healthy", {})
        else:
            g = generic.get("disease", {})
        description = g.get("description", f"Predicted class: {class_name}.")
        recommendations = g.get("recommendations", [
            "Isolate affected plants.",
            "Remove severely infected leaves and dispose safely.",
            "Consult local extension or plant pathology resources for treatment."
        ])
        localized_name = cname

    if not isinstance(recommendations, list):
        recommendations = list(recommendations) if recommendations else []
    description = str(description or "")
    localized_name = str(localized_name or cname)
    return bool(is_healthy), description, recommendations, localized_name

# Load model and classes (if present)
try:
    MODEL = safe_load_model(MODEL_PATH)
    CLASS_NAMES = load_class_names()
    INPUT_SHAPE = get_model_input_size(MODEL)
    logger.info(f"Model input size: {INPUT_SHAPE}")
except Exception:
    logger.exception("Initialization problem (model may be missing); /predict will return 503.")

@app.route("/", methods=["GET"])
def index():
    if os.path.exists(INDEX_HTML):
        return send_from_directory(BASE_DIR, "index.html")
    return (
        "<h3>Index not found.</h3>"
        "<p>Place your frontend HTML as <code>backend/index.html</code> and restart the server.</p>"
    )

@app.route("/i18n/<lang>.json", methods=["GET"])
def i18n(lang):
    lang_file = os.path.join(TRANSLATIONS_DIR, f"{lang}.json")
    if os.path.exists(lang_file):
        return send_from_directory(TRANSLATIONS_DIR, f"{lang}.json")
    return jsonify({})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if MODEL is not None else "model-not-loaded",
        "model_path": MODEL_PATH,
        "classes_count": len(CLASS_NAMES),
        "input_shape": {"height": INPUT_SHAPE[0], "width": INPUT_SHAPE[1], "channels": INPUT_SHAPE[2]}
    })

@app.route("/classes", methods=["GET"])
def classes():
    return jsonify({"classes": CLASS_NAMES})

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use key 'file' in multipart form."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        file.stream.seek(0)
        file.save(save_path)
        with open(save_path, "rb") as f:
            pil_img = decode_image_file(f)
    except Exception as e:
        logger.exception("Upload handling failed")
        return jsonify({"error": "Failed to read uploaded image.", "detail": str(e)}), 400

    try:
        x = preprocess_pil_image(pil_img, INPUT_SHAPE)
        preds = MODEL.predict(x)
        preds = np.array(preds).squeeze()

        if preds.ndim == 0:
            prob = float(preds)
            if CLASS_NAMES and len(CLASS_NAMES) == 2:
                idx = 1 if prob >= 0.5 else 0
                predicted_name = CLASS_NAMES[idx]
                confidence = prob if idx == 1 else 1 - prob
            else:
                predicted_name = CLASS_NAMES[1] if len(CLASS_NAMES) > 1 else ("Diseased" if prob >= 0.5 else "Healthy")
                confidence = prob
        else:
            exp = np.exp(preds - np.max(preds))
            probs = exp / np.sum(exp)
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            predicted_name = CLASS_NAMES[top_idx] if (CLASS_NAMES and top_idx < len(CLASS_NAMES)) else str(top_idx)

        # produce simplified key to help frontend translations
        simplified = simplify_class_key(predicted_name)

        # build generic localized info (default english) - note: frontend also has translations
        is_healthy, description, recommendations, localized_name = build_generic_disease_info(predicted_name, lang="en")

        return jsonify({
            "isHealthy": bool(is_healthy),
            "diseaseName": str(predicted_name),
            "diseaseName_simplified": str(simplified),
            "confidence": float(confidence),
            "description": description,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500

@app.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

if __name__ == "__main__":
    logger.info("Starting Flask app at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=DEBUG)
