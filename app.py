from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "blood_cancer_model.keras")

model = None

def load_model():
    global model
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# Class labels — must match training folder order (alphabetical)
# Update these to match your actual dataset folder names
CLASS_LABELS = {
    0: "Benign",
    1: "Early Pre-B ALL",
    2: "Pre-B ALL",
    3: "Pro-B ALL"
}

CLASS_DESCRIPTIONS = {
    0: "No cancerous cells detected. Blood cells appear normal.",
    1: "Early Pre-B Acute Lymphoblastic Leukemia detected.",
    2: "Pre-B Acute Lymphoblastic Leukemia detected.",
    3: "Pro-B Acute Lymphoblastic Leukemia detected."
}

CLASS_COLORS = {
    0: "benign",
    1: "early",
    2: "preb",
    3: "prob"
}

def preprocess_image(image_bytes):
    """Preprocess image exactly as done during training."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0  # rescale 1./255
    img_array = np.expand_dims(img_array, axis=0)        # add batch dim
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()

        # Preprocess
        img_array = preprocess_image(image_bytes)

        # Predict
        if model is None:
            return jsonify({"error": "Model not loaded. Please ensure blood_cancer_model.keras is in the /model folder."}), 500

        predictions = model.predict(img_array)[0]  # shape: (4,)
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        # Build result
        all_probs = {
            CLASS_LABELS[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(predictions))
        }

        # Encode image for preview
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        ext = file.content_type or "image/png"

        return jsonify({
            "predicted_class": CLASS_LABELS[predicted_class],
            "confidence": round(confidence, 2),
            "description": CLASS_DESCRIPTIONS[predicted_class],
            "color_class": CLASS_COLORS[predicted_class],
            "all_probabilities": all_probs,
            "image_data": f"data:{ext};base64,{img_b64}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()
    app.run(debug=True)
