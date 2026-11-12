
import sys
import sqlite3
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Locate the project root so imports work when launched from the BAT file.
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict


# --------------------------------------------------
# Configuration
# --------------------------------------------------

FRONTEND = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "deepleaf_cnn_final.keras"
UPLOAD = ROOT / "static" / "uploads"
UPLOAD.mkdir(parents=True, exist_ok=True)

DB = ROOT / "deepleaf.db"

app = Flask(
    __name__,
    static_folder=str(FRONTEND),
    static_url_path=""
)

CORS(app)

# Limit uploads to 10 MB.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# --------------------------------------------------
# Database
# --------------------------------------------------

def init_db():
    with sqlite3.connect(DB) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                filename TEXT,
                disease TEXT,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


init_db()


# --------------------------------------------------
# Frontend
# --------------------------------------------------

@app.get("/")
def home():
    return send_from_directory(str(FRONTEND), "index.html")


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_exists": MODEL_PATH.exists()
    })


# --------------------------------------------------
# Plant disease prediction
# --------------------------------------------------

@app.post("/api/predict")
def api_predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Image field is required."
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({
            "error": "Please select an image."
        }), 400

    # Validate the uploaded image before saving it.
    try:
        from PIL import Image

        uploaded_file.stream.seek(0)

        with Image.open(uploaded_file.stream) as image:
            image.verify()

        uploaded_file.stream.seek(0)

    except Exception:
        return jsonify({
            "error": "Invalid image. Upload a JPG, JPEG, or PNG file."
        }), 400

    if not MODEL_PATH.exists():
        return jsonify({
            "error": f"Trained model not found: {MODEL_PATH}"
        }), 500

    # Save with a generated filename instead of the user filename.
    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = UPLOAD / filename

    try:
        from PIL import Image

        uploaded_file.stream.seek(0)

        with Image.open(uploaded_file.stream) as image:
            image.convert("RGB").save(image_path, format="JPEG")

        result = predict(image_path)

        disease = result["disease"]
        confidence = float(result["confidence"])

        with sqlite3.connect(DB) as connection:
            connection.execute(
                """
                INSERT INTO predictions
                (id, filename, disease, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    filename,
                    disease,
                    confidence
                )
            )

        return jsonify(result)

    except Exception as error:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": f"Prediction failed: {error}"
        }), 500


# --------------------------------------------------
# Prediction history
# --------------------------------------------------

@app.get("/api/history")
def history():
    with sqlite3.connect(DB) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT id, filename, disease, confidence, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 50
        """).fetchall()

    return jsonify([dict(row) for row in rows])


# --------------------------------------------------
# Run server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
