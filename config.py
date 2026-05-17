import os

# Dynamic base directory mapping (works on both Windows and Railway Linux)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "val")
TEST_DIR    = os.path.join(DATASET_DIR, "test")

MODEL_DIR   = os.path.join(BASE_DIR, "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "plant_model.keras")
CLASS_JSON  = os.path.join(MODEL_DIR, "class_names.json")

OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
CM_DIR      = os.path.join(OUTPUT_DIR, "confusion_matrix")

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 25
NUM_CLASSES = 15   
LEARNING_RATE = 0.0001
