import os
import sys
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import *


def preprocess_image(img_path):

    img = Image.open(img_path).convert("RGB")

    img = img.resize(IMG_SIZE)

    img_array = np.array(img).astype(np.float32) / 255.0

    original_img = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_img


def overlay_heatmap(heatmap, original_img, alpha=0.4):

    h, w = original_img.shape[:2]

    heatmap = cv2.resize(heatmap, (w, h))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_img.astype(np.uint8),
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay


def run_gradcam(img_path, save_path=None):

    print("\n[INFO] Running Grad-CAM...")

    # --------------------------------
    # Load model
    # --------------------------------

    model = load_model(MODEL_PATH)

    print("[INFO] Model loaded")

    # --------------------------------
    # Load classes
    # --------------------------------

    with open(CLASS_JSON, "r") as f:
        class_names = json.load(f)

    # --------------------------------
    # Preprocess image
    # --------------------------------

    img_array, original_img = preprocess_image(img_path)

    # --------------------------------
    # Find nested base model
    # --------------------------------

    base_model = None

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            base_model = layer
            break

    if base_model is None:
        raise ValueError("Base model not found")

    print(f"[INFO] Base model: {base_model.name}")

    # --------------------------------
    # Find last Conv layer
    # --------------------------------

    last_conv_layer = None

    for layer in reversed(base_model.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):

            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found")

    print(f"[INFO] Last Conv Layer: {last_conv_layer.name}")

    # --------------------------------
    # Create temporary model
    # --------------------------------

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    # --------------------------------
    # Compute gradients
    # --------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    if grads is None:
        raise ValueError("Gradients are None")

    print("[INFO] Gradients computed")

    # --------------------------------
    # Generate heatmap
    # --------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = heatmap.numpy()

    # ReLU
    heatmap = np.maximum(heatmap, 0)

    # Normalize
    if np.max(heatmap) != 0:

        heatmap /= np.max(heatmap)

    # --------------------------------
    # Overlay heatmap
    # --------------------------------

    overlay = overlay_heatmap(
        heatmap,
        original_img
    )

    # --------------------------------
    # Save image
    # --------------------------------

    if save_path:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        Image.fromarray(
            overlay.astype(np.uint8)
        ).save(save_path)

        print(f"[INFO] Saved heatmap: {save_path}")

    # --------------------------------
    # Prediction info
    # --------------------------------

    confidence = float(
        predictions[0][pred_index]
    ) * 100

    class_name = (
        class_names[str(int(pred_index))]
        if isinstance(class_names, dict)
        else class_names[int(pred_index)]
    )

    print(f"[INFO] Prediction: {class_name}")
    print(f"[INFO] Confidence: {confidence:.2f}%")

    return overlay, class_name, confidence