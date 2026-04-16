#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a 3D CS-U-Net style classifier on 3D MRI volumes.

Expected volume layout:
    mri_3d_volumes/
        Training/<class_name>/vol_*.npy
        Testing/<class_name>/vol_*.npy

Each .npy file should be shape (D, H, W, 1) or (D, H, W).
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(split_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Load all .npy volumes from split folder grouped by class."""
    class_names = sorted(
        [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    )
    if not class_names:
        raise FileNotFoundError(f"No class directories found in: {split_dir}")

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    x, y = [], []

    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)
        files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(".npy")])
        print(f"[INFO] {os.path.basename(split_dir)}/{class_name}: {len(files)} volumes")
        for fname in files:
            path = os.path.join(class_dir, fname)
            vol = np.load(path)
            if vol.ndim == 3:
                vol = vol[..., np.newaxis]
            if vol.ndim != 4:
                print(f"[WARN] Skipping malformed volume: {path} (shape={vol.shape})")
                continue
            x.append(vol.astype(np.float32))
            y.append(class_to_idx[class_name])

    if not x:
        raise RuntimeError(f"No valid .npy volumes loaded from: {split_dir}")

    x = np.stack(x, axis=0)
    y = np.array(y, dtype=np.int32)
    return x, y, class_to_idx


def se_block_3d(inputs: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    """3D squeeze-and-excitation (channel attention)."""
    channels = int(inputs.shape[-1])
    se = layers.GlobalAveragePooling3D()(inputs)
    se = layers.Dense(max(channels // ratio, 1), activation="relu")(se)
    se = layers.Dense(channels, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, 1, channels))(se)
    return layers.Multiply()([inputs, se])


def conv_block_3d(x: tf.Tensor, filters: int, dropout: float = 0.0) -> tf.Tensor:
    x = layers.Conv3D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-5),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv3D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-5),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = se_block_3d(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def build_3d_csunet_classifier(
    input_shape: Tuple[int, int, int, int], num_classes: int
) -> Model:
    """3D CS-U-Net style encoder-decoder with classifier head."""
    inputs = layers.Input(shape=input_shape)

    # Encoder
    e1 = conv_block_3d(inputs, 16, dropout=0.05)
    p1 = layers.MaxPool3D(pool_size=2)(e1)

    e2 = conv_block_3d(p1, 32, dropout=0.1)
    p2 = layers.MaxPool3D(pool_size=2)(e2)

    e3 = conv_block_3d(p2, 64, dropout=0.15)
    p3 = layers.MaxPool3D(pool_size=2)(e3)

    # Bottleneck
    b = conv_block_3d(p3, 128, dropout=0.2)

    # Decoder (kept for U-Net style contextual fusion)
    u3 = layers.UpSampling3D(size=2)(b)
    u3 = layers.Concatenate()([u3, e3])
    d3 = conv_block_3d(u3, 64, dropout=0.1)

    u2 = layers.UpSampling3D(size=2)(d3)
    u2 = layers.Concatenate()([u2, e2])
    d2 = conv_block_3d(u2, 32, dropout=0.1)

    u1 = layers.UpSampling3D(size=2)(d2)
    u1 = layers.Concatenate()([u1, e1])
    d1 = conv_block_3d(u1, 16, dropout=0.05)

    # Classification head from fused decoder representation
    g = layers.GlobalAveragePooling3D()(d1)
    g = layers.Dense(64, activation="relu")(g)
    g = layers.Dropout(0.3)(g)
    outputs = layers.Dense(num_classes, activation="softmax")(g)

    model = Model(inputs=inputs, outputs=outputs, name="3D_CSU_Net_Classifier")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train 3D CS-U-Net classifier.")
    parser.add_argument(
        "--volumes-root",
        type=str,
        default="mri_3d_volumes",
        help="Root containing Training/Testing 3D .npy volumes.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="csunet_results")
    args = parser.parse_args()

    set_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    volumes_root = (
        args.volumes_root
        if os.path.isabs(args.volumes_root)
        else os.path.join(script_dir, args.volumes_root)
    )
    train_dir = os.path.join(volumes_root, "Training")
    test_dir = os.path.join(volumes_root, "Testing")

    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError(
            f"Could not find Training/Testing under volumes root: {volumes_root}"
        )

    output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(script_dir, args.output_dir)
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    x_train_all, y_train_all, class_to_idx = load_split(train_dir)
    x_test, y_test, class_to_idx_test = load_split(test_dir)

    if class_to_idx != class_to_idx_test:
        raise ValueError(
            "Class mismatch between Training and Testing splits. "
            f"Train map: {class_to_idx}, Test map: {class_to_idx_test}"
        )

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # Train/validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y_train_all,
    )

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    print(f"[INFO] x_train: {x_train.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}")
    print(f"[INFO] Classes: {class_to_idx}")

    # Build model
    input_shape = tuple(x_train.shape[1:])
    model = build_3d_csunet_classifier(input_shape=input_shape, num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    best_model_path = os.path.join(output_dir, "best_3d_csunet.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True),
    ]

    # Train
    history = model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    y_prob = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "class_to_idx": class_to_idx,
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=[idx_to_class[i] for i in range(num_classes)],
        output_dict=True,
    )

    # Save artifacts
    final_model_path = os.path.join(output_dir, "final_3d_csunet.keras")
    model.save(final_model_path)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    print("\n[OK] Training complete.")
    print(f"[OK] Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"[OK] Macro F1: {metrics['f1_macro']:.4f}")
    print(f"[OK] Saved best model: {best_model_path}")
    print(f"[OK] Saved final model: {final_model_path}")
    print(f"[OK] Metrics file: {os.path.join(output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
