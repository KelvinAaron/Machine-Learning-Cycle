import numpy as np
from sklearn import metrics as skm
from tensorflow import keras
from tensorflow.keras import layers

from src import config


def _augmentation():
    # Cells have no canonical orientation, so flips and rotations do not change the label.
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.25),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_cnn(cfg):
    inputs = keras.Input(shape=config.INPUT_SHAPE)
    x = _augmentation()(inputs) if cfg["augment"] else inputs
    x = layers.Rescaling(1 / 255)(x)
    if cfg["arch"] == "cnn_baseline":
        for filters in (32, 64, 128):
            x = layers.Conv2D(filters, 3, activation="relu")(x)
            x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
    else:
        for filters in (32, 64, 128):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(cfg["dropout"])(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=cfg["name"])


def build_mobilenet(cfg):
    base = keras.applications.MobileNetV2(
        input_shape=config.INPUT_SHAPE, include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = keras.Input(shape=config.INPUT_SHAPE)
    x = _augmentation()(inputs) if cfg["augment"] else inputs
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg["dropout"])(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=cfg["name"])


def build_model(cfg):
    return build_mobilenet(cfg) if cfg["arch"] == "mobilenetv2" else build_cnn(cfg)


def unfreeze_backbone(model, n_layers):
    base = next((layer for layer in model.layers if isinstance(layer, keras.Model)), None)
    # layers[:-0] is empty, which would leave the whole backbone trainable, so 0 must return early.
    if base is None or n_layers <= 0:
        return
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    # Batch statistics from small fine-tuning batches would destabilise the pretrained BN layers.
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def make_callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
    ]


def _merge_histories(histories):
    merged = {}
    for h in histories:
        for key, values in h.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def run_experiment(cfg, train_ds, val_ds):
    model = build_model(cfg)
    compile_model(model, cfg["lr"])
    histories = [
        model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"], callbacks=make_callbacks(), verbose=2)
    ]
    if cfg["unfreeze_last"] > 0:
        unfreeze_backbone(model, cfg["unfreeze_last"])
        compile_model(model, cfg["fine_tune_lr"])
        histories.append(
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg["fine_tune_epochs"],
                callbacks=make_callbacks(),
                verbose=2,
            )
        )
    return model, _merge_histories(histories)


def collect_predictions(model, ds):
    y_true, y_prob = [], []
    for x, y in ds:
        y_prob.append(model(x, training=False).numpy().ravel())
        y_true.append(y.numpy().ravel())
    return np.concatenate(y_true).astype(int), np.concatenate(y_prob)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(skm.roc_auc_score(y_true, y_prob)),
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_samples": int(len(y_true)),
    }


def evaluate_model(model, ds, threshold=0.5):
    y_true, y_prob = collect_predictions(model, ds)
    return compute_metrics(y_true, y_prob, threshold)


def load_model(path):
    return keras.models.load_model(path)
