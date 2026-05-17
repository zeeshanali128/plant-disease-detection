import os
import sys
import json
import shutil
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import *


def split_dataset():
    """
    Splits the raw PlantVillage dataset into train/val/test folders.
    Run this ONCE after placing dataset in dataset/raw/ folder.
    """
    RAW_DIR = os.path.join(DATASET_DIR, "raw")

    if not os.path.exists(RAW_DIR):
        print(f"[ERROR] Raw dataset not found at: {RAW_DIR}")
        print("Please place your PlantVillage dataset inside dataset/raw/")
        return

    classes = [d for d in os.listdir(RAW_DIR)
               if os.path.isdir(os.path.join(RAW_DIR, d))]
    print(f"[INFO] Found {len(classes)} classes.")

    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

    for cls in classes:
        cls_path = os.path.join(RAW_DIR, cls)
        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        n = len(images)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        splits = {
            'train': images[:train_end],
            'val':   images[train_end:val_end],
            'test':  images[val_end:]
        }

        for split, files in splits.items():
            for f in files:
                src = os.path.join(cls_path, f)
                dst = os.path.join(DATASET_DIR, split, cls, f)
                shutil.copy2(src, dst)

        print(f"  {cls}: {len(splits['train'])} train | "
              f"{len(splits['val'])} val | {len(splits['test'])} test")

    print("[DONE] Dataset split complete.")


def save_class_names(train_gen):
    os.makedirs(MODEL_DIR, exist_ok=True)
    class_names = list(train_gen.class_indices.keys())
    with open(CLASS_JSON, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"[INFO] Saved {len(class_names)} class names to {CLASS_JSON}")
    return class_names


def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen


if __name__ == "__main__":
    print("[STEP] Splitting dataset...")
    split_dataset()