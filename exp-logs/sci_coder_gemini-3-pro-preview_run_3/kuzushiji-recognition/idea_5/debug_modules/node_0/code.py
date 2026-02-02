import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import shutil

# Ensure reproducibility
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Import from the provided library files
from library.config import Config
from library.utils import (
    gaussian_radius,
    gaussian_2d,
    draw_gaussian,
    resize_with_padding,
    get_affine_transform,
    affine_transform,
    modified_f1_score,
)
from library.dataset import KuzushijiDetectionDataset, KuzushijiClassificationDataset
from library.models import CenterNetDetector, ResNetClassifier
from library.engine import run_training_and_inference


def main():
    print("=== Kuzushiji Pipeline Demonstration & Verification ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")
    # Override Config values to ensure the script runs very quickly
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 images
    Config.DETECTOR_EPOCHS = 1
    Config.CLASSIFIER_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure working directories exist
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORK_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # ------------------------------------------------------------------------
    # 2. Utility Functions Verification
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Gaussian Radius
    radius = gaussian_radius((100, 100), min_overlap=0.7)
    assert radius > 0, "Gaussian radius calculation failed"
    print(f"    gaussian_radius((100, 100)) -> {radius:.2f}")

    # Test Draw Gaussian
    heatmap = np.zeros((100, 100), dtype=np.float32)
    center = (50, 50)
    draw_gaussian(heatmap, center, radius=int(radius))
    assert np.isclose(heatmap[50, 50], 1.0), "Gaussian peak not at center"
    print("    draw_gaussian verified.")

    # Test Resize with Padding
    dummy_img = np.zeros((800, 600, 3), dtype=np.uint8)
    target_size = 1024
    padded, scale, pw, ph = resize_with_padding(dummy_img, target_size)
    assert padded.shape == (1024, 1024, 3), f"Resize shape mismatch: {padded.shape}"
    assert scale == 1024 / 800, "Scale calculation incorrect"
    print("    resize_with_padding verified.")

    # Test Affine Transform
    # Identity transform check
    center_pt = np.array([50, 50], dtype=np.float32)
    scale_s = 100.0
    rot = 0
    output_sz = [100, 100]
    trans = get_affine_transform(center_pt, scale_s, rot, output_sz)
    pt = np.array([50, 50], dtype=np.float32)
    new_pt = affine_transform(pt, trans)
    # With center at 50,50 and scale 100 mapping to 100x100, the center should map to 50,50
    assert np.allclose(new_pt, [50, 50], atol=1.0), f"Affine transform failed: {new_pt}"
    print("    affine_transform verified.")

    # Test Modified F1 Score
    # Case: Perfect match
    df_true = pd.DataFrame({"image_id": ["img1"], "labels": ["U+3000 10 10 20 20"]})
    df_pred = pd.DataFrame(
        {"image_id": ["img1"], "labels": ["U+3000 15 15"]}
    )  # Center (15,15) is inside (10,10,20,20)
    f1, p, r = modified_f1_score(df_true, df_pred)
    assert f1 == 1.0, f"F1 Score should be 1.0, got {f1}"
    print(f"    modified_f1_score verified (Score: {f1:.2f}).")

    # ------------------------------------------------------------------------
    # 3. Dataset Verification
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Datasets...")

    # Detection Dataset
    print("    Initializing KuzushijiDetectionDataset (Train)...")
    det_ds = KuzushijiDetectionDataset(split="train", debug=True)
    if len(det_ds) > 0:
        img_tensor, targets = det_ds[0]
        assert img_tensor.shape == (
            3,
            Config.DETECTOR_INPUT_SIZE,
            Config.DETECTOR_INPUT_SIZE,
        ), f"Detector input shape incorrect: {img_tensor.shape}"
        assert "hm" in targets and "wh" in targets, "Missing detector targets"
        print(f"    Detection sample verified. Input: {img_tensor.shape}")
    else:
        print("    Warning: Detection dataset is empty (check metadata).")

    # Classification Dataset
    print("    Initializing KuzushijiClassificationDataset (Train)...")
    # Force reload cache to ensure logic runs
    cache_file = os.path.join(Config.CACHE_DIR, "classifier_samples_train.npy")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    cls_ds = KuzushijiClassificationDataset(
        split="train", debug=True, load_cached_data=False
    )
    if len(cls_ds) > 0:
        crop_tensor, label = cls_ds[0]
        assert crop_tensor.shape == (
            3,
            Config.CLASSIFIER_INPUT_SIZE,
            Config.CLASSIFIER_INPUT_SIZE,
        ), f"Classifier input shape incorrect: {crop_tensor.shape}"
        assert isinstance(label.item(), int), "Label is not an integer"
        print(
            f"    Classification sample verified. Input: {crop_tensor.shape}, Label: {label}"
        )
    else:
        print("    Warning: Classification dataset is empty.")

    # ------------------------------------------------------------------------
    # 4. Model Verification
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Models...")

    # CenterNet Detector
    print("    Instantiating CenterNetDetector...")
    detector = CenterNetDetector(pretrained=False).to(Config.DEVICE)
    detector.eval()
    dummy_det_input = torch.randn(
        2, 3, Config.DETECTOR_INPUT_SIZE, Config.DETECTOR_INPUT_SIZE
    ).to(Config.DEVICE)
    with torch.no_grad():
        hm, wh, offset = detector(dummy_det_input)

    out_sz = Config.DETECTOR_INPUT_SIZE // 4
    assert hm.shape == (2, 1, out_sz, out_sz), f"Heatmap shape mismatch: {hm.shape}"
    assert wh.shape == (2, 2, out_sz, out_sz), f"Size head shape mismatch: {wh.shape}"
    print("    CenterNetDetector forward pass successful.")

    # ResNet Classifier
    print("    Instantiating ResNetClassifier...")
    classifier = ResNetClassifier(pretrained=False).to(Config.DEVICE)
    classifier.eval()
    dummy_cls_input = torch.randn(
        4, 3, Config.CLASSIFIER_INPUT_SIZE, Config.CLASSIFIER_INPUT_SIZE
    ).to(Config.DEVICE)
    with torch.no_grad():
        logits = classifier(dummy_cls_input)

    assert logits.shape == (
        4,
        Config.NUM_TOTAL_CLASSES,
    ), f"Classifier output mismatch: {logits.shape}"
    print("    ResNetClassifier forward pass successful.")

    # ------------------------------------------------------------------------
    # 5. Engine Execution (Training & Inference)
    # ------------------------------------------------------------------------
    print("\n[5] Running Full Pipeline (Mini-Train & Inference)...")

    # We use the provided engine function which handles training loops and inference
    # debug=True triggers the use of small datasets and fewer epochs (overridden in Config above)
    try:
        run_training_and_inference(debug=True)
        print("    Pipeline execution completed without errors.")
    except Exception as e:
        print(f"    Pipeline execution failed: {e}")
        raise e

    # ------------------------------------------------------------------------
    # 6. Output Validation
    # ------------------------------------------------------------------------
    print("\n[6] Validating Submission Output...")

    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"    Submission file found at {submission_path}")
        print(f"    Rows: {len(df_sub)}")
        print(f"    Columns: {list(df_sub.columns)}")

        # Check format
        assert (
            "image_id" in df_sub.columns and "labels" in df_sub.columns
        ), "Submission columns missing"

        # Check if we have predictions (might be empty if models are untrained/random, but structure should exist)
        # Note: With 1 epoch on 10 samples, predictions might be garbage or empty, but the code path is verified.
        print("    Submission format verified.")
    else:
        raise FileNotFoundError(f"Submission file not generated at {submission_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
