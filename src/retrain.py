import json
import os
import shutil
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src import model as M
from src import preprocessing as P


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_registry():
    if config.REGISTRY_PATH.exists():
        return json.loads(config.REGISTRY_PATH.read_text())
    return {"current": None, "versions": []}


def _save_registry(registry):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2))
    # Atomic swap: other containers poll this file and must never read a half-written one.
    tmp.replace(config.REGISTRY_PATH)


def current_entry(registry=None):
    registry = registry or load_registry()
    return next((v for v in registry["versions"] if v["version"] == registry["current"]), None)


def register_model(model, metrics, name, trigger="initial", n_new_images=0, promote=True):
    registry = load_registry()
    version = max((v["version"] for v in registry["versions"]), default=0) + 1
    filename = None
    if promote:
        filename = f"malaria_v{version}.keras"
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(config.MODELS_DIR / filename)
    registry["versions"].append(
        {
            "version": version,
            "name": name,
            "file": filename,
            "created_at": _now(),
            "trigger": trigger,
            "n_new_images": n_new_images,
            "promoted": promote,
            "metrics": metrics,
        }
    )
    if promote:
        registry["current"] = version
    _save_registry(registry)
    return version


def _count_images(class_name):
    class_dir = config.UPLOAD_DIR / class_name
    if not class_dir.exists():
        return 0
    return sum(1 for p in class_dir.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS)


def pending_uploads():
    return {c: _count_images(c) for c in config.CLASS_NAMES}


def should_retrain(threshold=None):
    return sum(pending_uploads().values()) >= (threshold or config.RETRAIN_THRESHOLD)


def is_not_worse(new, old, tolerance=config.PROMOTION_TOLERANCE):
    return new["f1"] >= old["f1"] - tolerance and new["recall"] >= old["recall"] - tolerance


def _consume_uploads(paths):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for src in paths:
        src = Path(src)
        dest_dir = config.TRAIN_DIR / src.parent.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dest_dir / f"up_{stamp}_{src.name}")


def run_retraining(trigger="manual", epochs=5, lr=1e-5, unfreeze_last=30):
    current = current_entry()
    if current is None:
        raise RuntimeError("No registered model to retrain.")
    # Snapshot first: files uploaded while training runs must stay pending for the next round.
    new_paths, new_labels = P.list_labelled_files(config.UPLOAD_DIR)
    if not new_paths:
        raise ValueError("No uploaded images to retrain on.")

    old_paths, old_labels = P.list_labelled_files(config.TRAIN_DIR)
    old_paths, old_labels = P.sample_files(old_paths, old_labels, config.OLD_SAMPLE_SIZE)
    paths, labels = old_paths + new_paths, old_labels + new_labels
    tr_p, tr_l, va_p, va_l = P.train_val_split(paths, labels, val_fraction=0.1)

    train_ds = P.make_dataset(tr_p, tr_l, shuffle=True, skip_errors=True)
    val_ds = P.make_dataset(va_p, va_l, skip_errors=True)
    test_ds = P.make_dataset(*P.list_labelled_files(config.TEST_DIR))

    model = M.load_model(config.MODELS_DIR / current["file"])
    # The registry scores may come from a larger test set than the one shipped here, so the live model is re-scored on the same data.
    baseline_metrics = M.evaluate_model(model, test_ds)
    M.unfreeze_backbone(model, unfreeze_last)
    M.compile_model(model, lr)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=M.make_callbacks(), verbose=2)

    metrics = M.evaluate_model(model, test_ds)
    promote = is_not_worse(metrics, baseline_metrics)
    version = register_model(
        model, metrics, current["name"], trigger=trigger, n_new_images=len(new_paths), promote=promote
    )
    _consume_uploads(new_paths)
    return {
        "version": version,
        "promoted": promote,
        "n_new_images": len(new_paths),
        "previous_version": current["version"],
        "previous_metrics": baseline_metrics,
        "new_metrics": metrics,
    }


def _idle_state():
    return {"status": "idle", "trigger": None, "started_at": None, "finished_at": None, "result": None, "error": None, "instance": None}


def read_state():
    if not config.RETRAIN_STATE_PATH.exists():
        return _idle_state()
    state = json.loads(config.RETRAIN_STATE_PATH.read_text())
    # A container that died mid-run leaves "running" behind; the lock file is the proof a run is still alive.
    if state["status"] == "running" and not config.RETRAIN_LOCK_PATH.exists():
        state = {**state, "status": "failed", "error": "Retraining was interrupted."}
    return state


def _write_state(state):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.RETRAIN_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(config.RETRAIN_STATE_PATH)


def _acquire_lock():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(config.RETRAIN_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        age = time.time() - config.RETRAIN_LOCK_PATH.stat().st_mtime
        if age < config.RETRAIN_LOCK_TIMEOUT:
            return False
        # Stale lock from a crashed container would otherwise block retraining forever.
        config.RETRAIN_LOCK_PATH.unlink(missing_ok=True)
        return _acquire_lock()


class RetrainJob:
    # State and lock live on the shared volume so every container sees one retraining run, not one each.
    @property
    def state(self):
        return read_state()

    def start(self, trigger="manual", **kwargs):
        if not _acquire_lock():
            return False
        _write_state({**_idle_state(), "status": "running", "trigger": trigger, "started_at": _now(), "instance": socket.gethostname()})
        threading.Thread(target=self._run, kwargs={"trigger": trigger, **kwargs}, daemon=True).start()
        return True

    def _run(self, **kwargs):
        try:
            update = {"status": "done", "result": run_retraining(**kwargs)}
        except Exception as exc:
            update = {"status": "failed", "error": str(exc)}
        try:
            _write_state({**read_state(), **update, "finished_at": _now()})
        finally:
            config.RETRAIN_LOCK_PATH.unlink(missing_ok=True)


job = RetrainJob()