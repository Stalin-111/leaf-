
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "deepleaf_cnn_final.keras"
CLASSES_PATH = PROJECT_DIR / "models" / "class_names.json"

IMG_SIZE = (160, 160)


def predict(path):
    """Predict the likely plant disease from an image file."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASSES_PATH}"
        )

    # Load model and class names
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    # Prepare image
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE)
        image_array = np.asarray(img, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    # The trained model contains a Rescaling layer,
    # so input pixel values must remain in the 0-255 range.
    probabilities = model.predict(image_array, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Model outputs and class names do not match."
        )

    best_index = int(np.argmax(probabilities))

    return {
        "disease": class_names[best_index],
        "confidence": round(
            float(probabilities[best_index]) * 100, 2
        ),
        "top_predictions": [
            {
                "disease": class_names[int(i)],
                "confidence": round(
                    float(probabilities[i]) * 100, 2
                ),
            }
            for i in np.argsort(probabilities)[::-1][:3]
        ],
    }
