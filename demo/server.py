#!/usr/bin/env python3
"""
Lightweight Flask server for the Pico-Banana diffusion image editor.
Wraps run_inference.py and serves the frontend.
"""

import os
import subprocess
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

app = Flask(__name__, static_folder=".")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/lora_step_5000.pt")
CONFIG = os.environ.get("CONFIG", "configs/training_config.yaml")


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/edit", methods=["POST"])
def edit():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    instruction = request.form.get("instruction", "").strip()
    if not instruction:
        return jsonify({"error": "No instruction provided."}), 400

    steps = int(request.form.get("steps", 50))
    seed = request.form.get("seed", "")

    job_id = uuid.uuid4().hex
    src_path = UPLOAD_DIR / f"{job_id}_src.png"
    out_path = OUTPUT_DIR / f"{job_id}_out.png"

    request.files["image"].save(src_path)

    cmd = [
        "python", "run_inference.py",
        "--source", str(src_path),
        "--instruction", instruction,
        "--checkpoint", CHECKPOINT,
        "--output", str(out_path),
        "--steps", str(steps),
        "--config", CONFIG,
    ]
    if seed:
        cmd += ["--seed", seed]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Inference timed out (>5 min)."}), 500

    if result.returncode != 0:
        return jsonify({"error": result.stderr or "Inference failed."}), 500

    return jsonify({"result_id": job_id})


@app.route("/result/<job_id>")
def result(job_id):
    # Sanitize
    safe_id = "".join(c for c in job_id if c.isalnum())
    out_path = OUTPUT_DIR / f"{safe_id}_out.png"
    if not out_path.exists():
        return jsonify({"error": "Result not found."}), 404
    return send_file(out_path, mimetype="image/png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    CHECKPOINT = args.checkpoint
    CONFIG = args.config

    print(f"Starting server on http://localhost:{args.port}")
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Config     : {CONFIG}")
    app.run(port=args.port, debug=False)
