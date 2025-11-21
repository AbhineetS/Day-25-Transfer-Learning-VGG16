#!/usr/bin/env python3
"""
Day 25 — Transfer Learning with VGG16 (fixed)
- Reads datasets with image_dataset_from_directory
- Ensures class_names is read BEFORE applying prefetch()/transformations
- Trains head, optionally fine-tunes last block
- Saves model and training plot
"""
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", "data")
BATCH_SIZE = 32
IMG_SIZE = (160, 160)   # modest size for speed
EPOCHS_HEAD = 6
EPOCHS_FINE = 4
MODEL_OUT = "vgg16_transfer.keras"
PLOT_OUT = "training_history_transfer.png"

def ensure_dummy_dataset():
    """Create a tiny dummy dataset if none exists (so script can run)."""
    base = Path(DATA_DIR)
    train_a = base / "train" / "class_a"
    train_b = base / "train" / "class_b"
    val_a = base / "val" / "class_a"
    val_b = base / "val" / "class_b"
    if base.exists():
        # looks like something exists — return
        return
    for p in (train_a, train_b, val_a, val_b):
        p.mkdir(parents=True, exist_ok=True)
    # small number of random images
    import numpy as _np
    for i in range(6):
        arr = (_np.random.rand(160,160,3)*255).astype("uint8")
        Image.fromarray(arr).save(train_a / f"img_a_{i}.jpg", quality=85)
    for i in range(6):
        arr = (_np.random.rand(160,160,3)*255).astype("uint8")
        Image.fromarray(arr).save(train_b / f"img_b_{i}.jpg", quality=85)
    for i in range(2):
        arr = (_np.random.rand(160,160,3)*255).astype("uint8")
        Image.fromarray(arr).save(val_a / f"val_a_{i}.jpg", quality=85)
    for i in range(2):
        arr = (_np.random.rand(160,160,3)*255).astype("uint8")
        Image.fromarray(arr).save(val_b / f"val_b_{i}.jpg", quality=85)
    print("�� Created tiny dummy dataset under", base)

def prepare_datasets(data_dir=DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Load train/val datasets and return (train_ds, val_ds, num_classes).

    Important: read class_names BEFORE applying prefetch/shuffle/etc.
    """
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    if not os.path.isdir(train_path) or not os.path.isdir(val_path):
        print("⚠️ data/train or data/val not found — creating a tiny dummy dataset so script can run.")
        ensure_dummy_dataset()

    # load raw datasets (do not transform yet)
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        val_path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    # read class names BEFORE transforming (prefetch/augment) — attribute available on raw ds
    try:
        class_names = list(train_ds_raw.class_names)
    except Exception as e:
        # fallback: infer from directory listing
        class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    print("✔ Classes:", class_names)

    # now apply performance transforms
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.prefetch(AUTOTUNE)
    val_ds = val_ds_raw.prefetch(AUTOTUNE)

    return train_ds, val_ds, len(class_names)

def build_model(num_classes, input_shape=IMG_SIZE + (3,)):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # freeze for head training

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="vgg16_transfer")
    return model, base_model

def compile_and_train(model, train_ds, val_ds, epochs):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=2)
    return history

def fine_tune(model, base_model, train_ds, val_ds, fine_epochs=EPOCHS_FINE):
    # unfreeze last VGG block (block5) for gentle fine-tuning
    for layer in base_model.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    history_ft = model.fit(train_ds, epochs=fine_epochs, validation_data=val_ds, verbose=2)
    return history_ft

def save_artifacts(model, history, history_ft=None):
    try:
        model.save(MODEL_OUT)
        print(f"💾 Saved model: {MODEL_OUT}")
    except Exception as e:
        print("⚠️ Could not save model:", e)

    # Combine histories for plotting
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    if history_ft:
        acc = acc + history_ft.history.get("accuracy", [])
        val_acc = val_acc + history_ft.history.get("val_accuracy", [])

    try:
        plt.figure()
        plt.plot(acc, label="train")
        plt.plot(val_acc, label="val")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(PLOT_OUT, dpi=150)
        plt.close()
        print(f"💾 Saved training plot: {PLOT_OUT}")
    except Exception as e:
        print("⚠️ Could not save training plot:", e)

def main():
    train_ds, val_ds, num_classes = prepare_datasets()
    model, base_model = build_model(num_classes)
    print("🧠 Model summary:")
    model.summary()

    print("🚀 Training head...")
    history = compile_and_train(model, train_ds, val_ds, EPOCHS_HEAD)

    print("🔧 Fine-tuning last block...")
    history_ft = fine_tune(model, base_model, train_ds, val_ds)

    save_artifacts(model, history, history_ft)
    print("✅ Done.")

if __name__ == "__main__":
    main()
