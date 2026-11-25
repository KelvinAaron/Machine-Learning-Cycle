import threading
import time

import tensorflow as tf

from src import config
from src import model as M
from src import preprocessing as P
from src.retrain import current_entry, load_registry


class ModelNotAvailable(Exception):
    pass


class InvalidImage(Exception):
    pass


class Predictor:
    def __init__(self):
        self._model = None
        self._version = None
        self._registry_mtime = None
        self._lock = threading.Lock()

    @property
    def version(self):
        return self._version

    def _registry_stamp(self):
        return config.REGISTRY_PATH.stat().st_mtime if config.REGISTRY_PATH.exists() else None

    def _ensure_current(self):
        stamp = self._registry_stamp()
        if self._model is not None and stamp == self._registry_mtime:
            return
        with self._lock:
            entry = current_entry(load_registry())
            if entry is None or entry["file"] is None:
                raise ModelNotAvailable("No trained model is registered yet.")
            # Reload only when the promoted version changed; another container may have retrained.
            if entry["version"] != self._version:
                self._model = M.load_model(config.MODELS_DIR / entry["file"])
                self._version = entry["version"]
            self._registry_mtime = stamp

    def predict(self, image_bytes):
        self._ensure_current()
        start = time.perf_counter()
        try:
            image = P.decode_resize(image_bytes)
        except Exception as exc:
            raise InvalidImage("File is not a readable PNG, JPEG or BMP image.") from exc
        prob = float(self._model(tf.expand_dims(image, 0), training=False).numpy()[0][0])
        label = config.CLASS_NAMES[int(prob >= 0.5)]
        return {
            "label": label,
            "probability_parasitized": round(prob, 4),
            "confidence": round(prob if label == "parasitized" else 1 - prob, 4),
            "model_version": self._version,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }


predictor = Predictor()
