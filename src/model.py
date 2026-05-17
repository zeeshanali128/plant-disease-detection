import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import NUM_CLASSES, LEARNING_RATE, IMG_SIZE


def build_model(num_classes=NUM_CLASSES, fine_tune=False):
    """
    Builds a transfer learning model using MobileNetV2 base.
    Phase 1: base frozen  (fast initial training)
    Phase 2: fine_tune=True unfreezes top 30 layers
    """
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )

    if fine_tune:
        # Unfreeze top 30 layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        print(f"[INFO] Fine-tuning: top 30 layers of MobileNetV2 unfrozen.")
    else:
        base_model.trainable = False
        print("[INFO] Base model frozen for transfer learning phase.")

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    lr = LEARNING_RATE if not fine_tune else LEARNING_RATE / 10
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"[INFO] Model compiled | LR={lr} | Classes={num_classes}")
    model.summary()
    return model


if __name__ == "__main__":
    m = build_model()
    print("Model built successfully.")