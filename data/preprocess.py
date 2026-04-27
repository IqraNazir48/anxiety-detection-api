# data/preprocess.py
# Phase 1 — Dataset Preprocessing Script
# Reads FER2013 from image folders (train/test) and CK+ from CK+48/ck folders

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Emotion labels ─────────────────────────────────────────────────────────
EMOTION_LABELS   = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_INDEX = {name: i for i, name in enumerate(EMOTION_LABELS)}

# ══════════════════════════════════════════════════════════════════════════
# HELPER — load all images from a folder tree like train/angry/*.jpg
# ══════════════════════════════════════════════════════════════════════════

def load_image_folder(root_dir):
    images, labels = [], []
    for emotion_name in EMOTION_LABELS:
        folder = os.path.join(root_dir, emotion_name)
        if not os.path.exists(folder):
            print(f"  [WARN] Folder not found, skipping: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"    [{EMOTION_TO_INDEX[emotion_name]}] {emotion_name:10s}: {len(files)} images")
        for fname in files:
            img_path = os.path.join(folder, fname)
            try:
                img = Image.open(img_path).convert("L").resize((48, 48))
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(EMOTION_TO_INDEX[emotion_name])
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")
    return np.array(images), np.array(labels)


# ══════════════════════════════════════════════════════════════════════════
# PART 1 — FER2013
# ══════════════════════════════════════════════════════════════════════════

def process_fer2013():
    print("\n" + "="*55)
    print("  PART 1: Processing FER2013 Dataset")
    print("="*55)