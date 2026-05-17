import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import *
from src.data_preparation import get_data_generators


def evaluate_model():
    print("\n" + "="*60)
    print("  PLANT DISEASE DETECTION — EVALUATION")
    print("="*60)

    # ── Load model & class names ───────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Please run train.py first.")
        return

    print(f"\n[INFO] Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    with open(CLASS_JSON, 'r') as f:
        class_names = json.load(f)

    # ── Get test generator ─────────────────────────────────────
    _, _, test_gen = get_data_generators()
    test_gen.reset()

    print(f"[INFO] Evaluating on {test_gen.samples} test images...")

    # ── Predictions ────────────────────────────────────────────
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # ── Accuracy ───────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[RESULT] Test Accuracy: {acc*100:.2f}%")

    # ── Classification report ──────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    print("\n[CLASSIFICATION REPORT]")
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)
    print(f"[INFO] Report saved: {report_path}")

    # ── Confusion matrix ───────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(CM_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(28, 24))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(f'Confusion Matrix — Test Accuracy: {acc*100:.2f}%',
                 fontsize=16, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()

    cm_path = os.path.join(CM_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved: {cm_path}")

    # ── Top-5 worst predicted classes ─────────────────────────
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    worst_idx = np.argsort(per_class_acc)[:5]
    print("\n[INFO] Top-5 hardest classes:")
    for i in worst_idx:
        print(f"  {class_names[i]:<45} acc={per_class_acc[i]*100:.1f}%")

    return acc, report


if __name__ == "__main__":
    evaluate_model()