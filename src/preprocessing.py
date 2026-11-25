import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config


def _download(url, dest):
    # Default urllib headers get blocked by some servers, which then answer with an HTML page instead of the file.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


def download_dataset(raw_dir=config.RAW_DIR):
    raw_dir = Path(raw_dir)
    if any(raw_dir.rglob("*.png")):
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "cell_images.zip"
    archive.unlink(missing_ok=True)
    for url in config.DATASET_URLS:
        try:
            _download(url, archive)
            # A server can return 200 with an error page, so the content is checked, not the status.
            if zipfile.is_zipfile(archive):
                break
            print(f"{url} did not return a zip file")
        except Exception as exc:
            print(f"download failed from {url}: {exc}")
    else:
        archive.unlink(missing_ok=True)
        raise RuntimeError("Could not download the dataset; place cell_images.zip in data/raw manually.")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    archive.unlink()
    return raw_dir


def _find_class_dir(raw_dir, class_name):
    candidates = [p for p in Path(raw_dir).rglob("*") if p.is_dir() and p.name.lower() == class_name]
    # The archive can contain a nested duplicate cell_images folder; take the one that holds the images.
    return max(candidates, key=lambda p: len(list(p.glob("*.png"))))


def split_dataset(raw_dir=config.RAW_DIR, data_dir=config.DATA_DIR, test_fraction=0.2, seed=config.SEED):
    data_dir = Path(data_dir)
    if any((data_dir / "train").rglob("*.png")):
        return
    rng = random.Random(seed)
    for class_name in config.CLASS_NAMES:
        # glob("*.png") also skips the stray Thumbs.db files shipped in the archive.
        files = sorted(_find_class_dir(raw_dir, class_name).glob("*.png"))
        rng.shuffle(files)
        n_test = int(len(files) * test_fraction)
        for subset, subset_files in (("test", files[:n_test]), ("train", files[n_test:])):
            dest = data_dir / subset / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for f in subset_files:
                shutil.copy2(f, dest / f.name)


def list_labelled_files(directory):
    paths, labels = [], []
    for label, class_name in enumerate(config.CLASS_NAMES):
        class_dir = Path(directory) / class_name
        if not class_dir.exists():
            continue
        files = sorted(str(p) for p in class_dir.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS)
        paths += files
        labels += [label] * len(files)
    return paths, labels


def sample_files(paths, labels, n, seed=config.SEED):
    if len(paths) <= n:
        return list(paths), list(labels)
    idx = random.Random(seed).sample(range(len(paths)), n)
    return [paths[i] for i in idx], [labels[i] for i in idx]


def train_val_split(paths, labels, val_fraction=0.15, seed=config.SEED):
    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=val_fraction, stratify=labels, random_state=seed
    )
    return tr_p, tr_l, va_p, va_l


def decode_resize(raw):
    # Training and serving share this exact function so resizing can never drift between them.
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, config.IMG_SIZE)


def _load(path, label):
    img = decode_resize(tf.io.read_file(path))
    return img, tf.cast(label, tf.float32)[None]


def make_dataset(paths, labels, batch_size=config.BATCH_SIZE, shuffle=False, seed=config.SEED, skip_errors=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if skip_errors:
        # A corrupt user upload must not abort a retraining run; it makes the dataset length unknown, so only used there.
        ds = ds.ignore_errors()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def image_features(path):
    with Image.open(path) as im:
        width, height = im.size
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    # Cells sit on a black background; features are measured on cell pixels only.
    cell = rgb.max(axis=2) > config.BACKGROUND_THRESHOLD
    if not cell.any():
        cell = np.ones(cell.shape, dtype=bool)
    px = rgb[cell]
    mx, mn = px.max(axis=1), px.min(axis=1)
    mean_r, mean_g, mean_b = px.mean(axis=0)
    return {
        "width": width,
        "height": height,
        "cell_fraction": float(cell.mean()),
        "mean_r": float(mean_r),
        "mean_g": float(mean_g),
        "mean_b": float(mean_b),
        "saturation": float(((mx - mn) / np.maximum(mx, 1)).mean()),
        # Giemsa stain absorbs green light, so low-green pixels mark stained material.
        "stain_fraction": float((px[:, 1] < config.STAIN_GREEN_THRESHOLD).mean()),
    }
