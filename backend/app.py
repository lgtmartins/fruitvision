"""
FruitVision - backend (API REST).

Carrega o modelo YOLO treinado pelo grupo (training/train.py) e expõe uma
API HTTP simples para que o frontend (ou qualquer cliente) envie uma
imagem e receba de volta as frutas detectadas, junto com a imagem
anotada. Cada detecção é salva no banco SQLite (ver database/db.py).

Nenhuma API de IA generativa/externa é usada: a "inteligência" aqui é o
modelo YOLO treinado localmente pelo grupo com o dataset próprio.

Como rodar:
    pip install -r backend/requirements.txt
    python backend/app.py

A API sobe em http://localhost:5000
"""

import os
import sys
import uuid
from pathlib import Path

import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from database.db import (  # noqa: E402
    init_db,
    save_detection,
    get_history,
    get_detection,
    get_stats,
    delete_detection,
)

# Pode ser trocado pela variável de ambiente FRUITVISION_WEIGHTS (ex.: para
# testar o modelo pré-treinado sem mexer no código).
WEIGHTS = Path(os.environ.get(
    "FRUITVISION_WEIGHTS",
    ROOT / "runs" / "fruitvision_yolo26n_scratch" / "weights" / "best.pt",
))
UPLOAD_DIR = ROOT / "backend" / "uploads"
RESULTS_DIR = ROOT / "backend" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15 MB por upload

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

# Garante que as tabelas existem independentemente de como o app é iniciado
# (python backend/app.py, flask run, gunicorn, etc.)
init_db()

_model = None


def get_model():
    """Carrega o modelo sob demanda (uma única vez) e o mantém em memória."""
    global _model
    if _model is None:
        if not WEIGHTS.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {WEIGHTS}.\n"
                "Treine o modelo primeiro com: python training/train.py"
            )
        _model = YOLO(str(WEIGHTS))
    return _model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "weights_found": WEIGHTS.exists(),
        "model_loaded": _model is not None,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada. Envie um arquivo no campo 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nome de arquivo vazio."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Formato não suportado. Use .jpg, .jpeg ou .png."}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    try:
        conf_threshold = float(request.form.get("conf", 0.25))
    except ValueError:
        conf_threshold = 0.25
    conf_threshold = max(0.01, min(conf_threshold, 0.99))

    unique_id = uuid.uuid4().hex[:10]
    extension = file.filename.rsplit(".", 1)[1].lower()
    upload_path = UPLOAD_DIR / f"{unique_id}.{extension}"
    file.save(upload_path)

    try:
        results = model.predict(source=str(upload_path), conf=conf_threshold, save=False, verbose=False)
    except Exception as e:
        return jsonify({"error": f"Falha ao processar a imagem: {e}"}), 500

    result = results[0]

    annotated = result.plot()  # numpy array (BGR) com as caixas desenhadas
    result_filename = f"{unique_id}_result.jpg"
    result_path = RESULTS_DIR / result_filename
    cv2.imwrite(str(result_path), annotated)

    items = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
            items.append({
                "class_name": result.names[cls_id],
                "confidence": round(confidence, 4),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    detection_id = save_detection(
        original_filename=file.filename,
        annotated_image_path=f"results/{result_filename}",
        conf_threshold=conf_threshold,
        items=items,
    )

    return jsonify({
        "detection_id": detection_id,
        "original_filename": file.filename,
        "conf_threshold": conf_threshold,
        "annotated_image_url": f"/api/images/results/{result_filename}",
        "detections": items,
        "total_objects": len(items),
    })


@app.route("/api/images/<subfolder>/<filename>", methods=["GET"])
def serve_image(subfolder, filename):
    if subfolder == "uploads":
        directory = UPLOAD_DIR
    elif subfolder == "results":
        directory = RESULTS_DIR
    else:
        return jsonify({"error": "Pasta inválida."}), 404
    return send_from_directory(directory, filename)


@app.route("/api/history", methods=["GET"])
def history():
    limit = request.args.get("limit", default=50, type=int)
    return jsonify(get_history(limit))


@app.route("/api/history/<int:detection_id>", methods=["GET"])
def history_detail(detection_id):
    detection = get_detection(detection_id)
    if detection is None:
        return jsonify({"error": "Detecção não encontrada."}), 404
    return jsonify(detection)


@app.route("/api/history/<int:detection_id>", methods=["DELETE"])
def history_delete(detection_id):
    deleted = delete_detection(detection_id)
    if not deleted:
        return jsonify({"error": "Detecção não encontrada."}), 404
    return jsonify({"deleted": True})


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(get_stats())


if __name__ == "__main__":
    print(f"Pesos do modelo: {WEIGHTS} ({'encontrado' if WEIGHTS.exists() else 'NÃO encontrado - treine antes com training/train.py'})")
    app.run(host="0.0.0.0", port=5000, debug=True)
