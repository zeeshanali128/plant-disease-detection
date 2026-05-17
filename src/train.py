import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import *
from src.data_preparation import get_data_generators, save_class_names
from src.model import build_model


def plot_history(history, prefix="phase1"):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'],     label='Train Acc',  color='#2196F3')
    ax1.plot(history.history['val_accuracy'], label='Val Acc',    color='#4CAF50')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train Loss', color='#F44336')
    ax2.plot(history.history['val_loss'], label='Val Loss',   color='#FF9800')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"{prefix}_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Plot saved: {save_path}")


def get_callbacks(checkpoint_path):
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  PLANT DISEASE DETECTION — FULL TRAINING")
    print("="*60)

    # ── Load data ──────────────────────────────────────────────
    print("\n[STEP 1] Loading data generators...")
    train_gen, val_gen, test_gen = get_data_generators()
    save_class_names(train_gen)

    actual_classes = train_gen.num_classes
    print(f"[INFO] Classes found  : {actual_classes}")
    print(f"[INFO] Train samples  : {train_gen.samples}")
    print(f"[INFO] Val samples    : {val_gen.samples}")
    print(f"[INFO] Test samples   : {test_gen.samples}")

    # ── Phase 1: Frozen base ───────────────────────────────────
    print("\n[STEP 2] Phase 1 — Training with frozen MobileNetV2 base...")
    model = build_model(num_classes=actual_classes, fine_tune=False)
    best_ckpt = os.path.join(MODEL_DIR, "best_phase1.keras")

    history1 = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(best_ckpt),
        verbose=1
    )
    plot_history(history1, prefix="phase1")

    # Load best Phase 1 weights back into same model
    model.load_weights(best_ckpt)
    p1_loss, p1_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n[Phase 1 Result] Val Accuracy: {p1_acc*100:.2f}%")

    # ── Phase 2: Unfreeze top layers of SAME model ─────────────
    print("\n[STEP 3] Phase 2 — Fine-tuning top 30 layers...")

    # Get the MobileNetV2 base layer from the same model
    base_model = model.layers[1]  # MobileNetV2 is layer index 1
    base_model.trainable = True

    # Freeze all except last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[INFO] Unfrozen MobileNetV2 layers: {trainable_count}")

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    best_ft_ckpt = os.path.join(MODEL_DIR, "best_finetune.keras")

    history2 = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=get_callbacks(best_ft_ckpt),
        verbose=1
    )
    plot_history(history2, prefix="phase2_finetune")

    # Load best fine-tuned weights
    model.load_weights(best_ft_ckpt)

    # ── Save final model ───────────────────────────────────────
    final_path = os.path.join(MODEL_DIR, "plant_model.keras")
    model.save(final_path)
    print(f"\n[DONE] Final model saved: {final_path}")

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"[RESULT] Val Accuracy : {val_acc*100:.2f}%")
    print(f"[RESULT] Val Loss     : {val_loss:.4f}")

    return model, train_gen, val_gen, test_gen


if __name__ == "__main__":
    train()