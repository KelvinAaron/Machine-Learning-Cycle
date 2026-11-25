import io
import json
import socket
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from src import config
from src import preprocessing as P
from src.prediction import InvalidImage, ModelNotAvailable, predictor
from src.retrain import current_entry, job, load_registry, pending_uploads, should_retrain

@asynccontextmanager
async def lifespan(_):
    try:
        predictor.warm_up()
    except ModelNotAvailable:
        pass
    yield


app = FastAPI(title="Malaria Cell Classifier API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"]
)

STARTED_AT = time.time()
counters = {"requests": 0, "predictions": 0, "errors": 0}
recent_latencies = deque(maxlen=200)


@app.middleware("http")
async def count_requests(request, call_next):
    counters["requests"] += 1
    response = await call_next(request)
    if response.status_code >= 500:
        counters["errors"] += 1
    return response


class RetrainRequest(BaseModel):
    epochs: int = Field(5, ge=1, le=20)
    lr: float = Field(1e-5, ge=1e-7, le=1e-3)
    unfreeze_last: int = Field(30, ge=0, le=150)


def _version():
    entry = current_entry(load_registry())
    return entry["version"] if entry else None


def _read_limited(upload):
    data = upload.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{upload.filename} exceeds {config.MAX_UPLOAD_BYTES // 1024 // 1024} MB.")
    return data


def _is_valid_image(data):
    try:
        with Image.open(io.BytesIO(data)) as im:
            # Restricted to formats TensorFlow's decoder also reads, so a stored file never breaks training.
            if im.format not in {"PNG", "JPEG", "BMP"}:
                return False
            im.verify()
        return True
    except Exception:
        return False


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "started_at": _iso(STARTED_AT),
        "model_version": _version(),
        "instance": socket.gethostname(),
    }


@app.get("/status")
def status():
    lat = np.array(recent_latencies) if recent_latencies else None
    return {
        **health(),
        "counters": counters,
        "latency_ms": {
            "mean": round(float(lat.mean()), 2) if lat is not None else None,
            "p95": round(float(np.percentile(lat, 95)), 2) if lat is not None else None,
            "samples": len(recent_latencies),
        },
        "pending_uploads": pending_uploads(),
        "retrain_threshold": config.RETRAIN_THRESHOLD,
        "auto_retrain": config.AUTO_RETRAIN,
        "retrain": job.state,
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = _read_limited(file)
    try:
        result = predictor.predict(data)
    except ModelNotAvailable as exc:
        raise HTTPException(503, str(exc))
    except InvalidImage as exc:
        raise HTTPException(400, str(exc))
    counters["predictions"] += 1
    recent_latencies.append(result["latency_ms"])
    return result


@app.post("/upload")
def upload(label: str = Form(...), files: List[UploadFile] = File(...)):
    if label not in config.CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {config.CLASS_NAMES}.")
    dest = config.UPLOAD_DIR / label
    dest.mkdir(parents=True, exist_ok=True)
    saved, rejected = 0, []
    for f in files:
        data = _read_limited(f)
        if not _is_valid_image(data):
            rejected.append(f.filename)
            continue
        suffix = "." + (Image.open(io.BytesIO(data)).format or "PNG").lower().replace("jpeg", "jpg")
        (dest / f"{uuid.uuid4().hex}{suffix}").write_bytes(data)
        saved += 1
    auto_started = False
    if config.AUTO_RETRAIN and should_retrain():
        auto_started = job.start(trigger="auto")
    return {
        "saved": saved,
        "rejected": rejected,
        "pending_uploads": pending_uploads(),
        "auto_retrain_started": auto_started,
    }


@app.post("/retrain", status_code=202)
def retrain(params: RetrainRequest = RetrainRequest()):
    if sum(pending_uploads().values()) == 0:
        raise HTTPException(400, "Upload new images before retraining.")
    if not job.start(trigger="manual", **params.model_dump()):
        raise HTTPException(409, "A retraining run is already in progress.")
    return {"started": True}


@app.get("/retrain/status")
def retrain_status():
    return job.state


@app.get("/metrics")
def metrics():
    registry = load_registry()
    return {"current": registry["current"], "versions": registry["versions"]}


def _count_dir(directory):
    return {c: sum(1 for p in (directory / c).glob("*") if p.suffix.lower() in config.IMAGE_EXTENSIONS)
            if (directory / c).exists() else 0 for c in config.CLASS_NAMES}


@app.get("/data/summary")
def data_summary():
    return {
        "train": _count_dir(config.TRAIN_DIR),
        "test": _count_dir(config.TEST_DIR),
        "uploads": _count_dir(config.UPLOAD_DIR),
    }


@app.get("/data/features")
def data_features(per_class: int = 150):
    train_counts = _count_dir(config.TRAIN_DIR)
    key = {"train": train_counts, "per_class": per_class}
    if config.FEATURE_CACHE_PATH.exists():
        cached = json.loads(config.FEATURE_CACHE_PATH.read_text())
        if cached.get("key") == key:
            return cached["payload"]

    bins = np.linspace(0, 1, 11)
    classes = {}
    paths, labels = P.list_labelled_files(config.TRAIN_DIR)
    for label, name in enumerate(config.CLASS_NAMES):
        class_paths = [p for p, l in zip(paths, labels) if l == label]
        if not class_paths:
            continue
        sample, _ = P.sample_files(class_paths, [label] * len(class_paths), per_class)
        feats = [P.image_features(p) for p in sample]
        col = lambda k: np.array([f[k] for f in feats])
        classes[name] = {
            "mean_rgb": [round(float(col(k).mean()), 2) for k in ("mean_r", "mean_g", "mean_b")],
            "mean_saturation": round(float(col("saturation").mean()), 4),
            "mean_stain_fraction": round(float(col("stain_fraction").mean()), 4),
            "stain_hist": np.histogram(col("stain_fraction"), bins=bins)[0].tolist(),
            "mean_width": round(float(col("width").mean()), 1),
            "mean_height": round(float(col("height").mean()), 1),
            "sample_size": len(feats),
        }
    payload = {"stain_bins": [round(float(b), 2) for b in bins], "classes": classes}
    config.FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.FEATURE_CACHE_PATH.write_text(json.dumps({"key": key, "payload": payload}))
    return payload