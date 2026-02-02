import os
import sys
import pandas as pd
import torch
import numpy as np
import cv2

# Import from the provided library
from library.config import Config, seed_everything
from library.data import get_dataloaders, SegmentationDataset, ClassificationDataset
from library.models import SegmentationUNet, CharacterClassifier
from library.train import train_segmenter, train_classifier
from library.inference import generate_submission, load_models, process_page
from library.utils import load_image


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("=== Setting up Configuration for Demo ===")
    seed_everything(Config.SEED)

    # Override Config parameters to run a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 images for training/debugging
    Config.SEG_EPOCHS = 1
    Config.CLS_EPOCHS = 1
    Config.SEG_BATCH_SIZE = 4
    Config.CLS_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small script

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")
    print(f"Cache Directory: {Config.CACHE_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration & Verification
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating Data Loading ===")

    # 2.1 Segmentation DataLoader
    print("Loading Segmentation Data (Debug Mode)...")
    seg_loader = get_dataloaders("segmentation", "train", debug=True)

    # Fetch one batch to verify
    seg_images, seg_masks = next(iter(seg_loader))

    print(f"Segmentation Batch - Images: {seg_images.shape}, Masks: {seg_masks.shape}")

    # Assertions for Segmentation
    assert seg_images.dim() == 4, "Segmentation images should be 4D (B, C, H, W)"
    assert seg_images.shape[1] == 3, "Segmentation images should have 3 channels (RGB)"
    assert (
        seg_masks.dim() == 3
    ), "Segmentation masks should be 3D (B, H, W) for CrossEntropy/Long or similar"
    # Note: The Dataset returns mask as (H, W), loader stacks to (B, H, W).
    # The model expects (B, 1, H, W) or (B, H, W) depending on loss.
    # train.py unsqueezes it to (B, 1, H, W).

    # 2.2 Classification DataLoader
    print("Loading Classification Data (Debug Mode)...")
    # This triggers cache generation for classification metadata
    cls_loader = get_dataloaders("classification", "train", debug=True)

    if len(cls_loader) > 0:
        cls_images, cls_labels = next(iter(cls_loader))
        print(
            f"Classification Batch - Images: {cls_images.shape}, Labels: {cls_labels.shape}"
        )

        # Assertions for Classification
        assert cls_images.dim() == 4, "Classification images should be 4D"
        assert (
            cls_images.shape[2:] == Config.CLS_CROP_SIZE
        ), f"Images should be resized to {Config.CLS_CROP_SIZE}"
        assert cls_labels.dim() == 1, "Labels should be 1D tensor"
    else:
        print("Warning: No characters found in the debug sample for classification.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass Verification
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating Model Instantiation ===")

    # 3.1 Segmentation Model
    seg_model = SegmentationUNet(n_classes=1).to(Config.DEVICE)
    dummy_seg_input = torch.randn(
        2, 3, Config.SEG_IMG_SIZE[0], Config.SEG_IMG_SIZE[1]
    ).to(Config.DEVICE)
    with torch.no_grad():
        seg_out = seg_model(dummy_seg_input)

    print(f"Segmentation Model Output Shape: {seg_out.shape}")
    assert seg_out.shape == (
        2,
        1,
        Config.SEG_IMG_SIZE[0],
        Config.SEG_IMG_SIZE[1],
    ), "Segmentation output shape mismatch"

    # 3.2 Classification Model
    # Determine number of classes from the dataset loaded earlier
    num_classes = len(cls_loader.dataset.label2id)
    print(f"Number of classes detected: {num_classes}")

    cls_model = CharacterClassifier(num_classes=num_classes).to(Config.DEVICE)
    dummy_cls_input = torch.randn(
        2, 3, Config.CLS_CROP_SIZE[0], Config.CLS_CROP_SIZE[1]
    ).to(Config.DEVICE)
    with torch.no_grad():
        cls_out = cls_model(dummy_cls_input)

    print(f"Classification Model Output Shape: {cls_out.shape}")
    assert cls_out.shape == (2, num_classes), "Classification output shape mismatch"

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating Training Loops ===")

    # 4.1 Train Segmentation Model
    print("Training Segmenter (1 Epoch)...")
    seg_model_path = train_segmenter(debug=True, epochs=Config.SEG_EPOCHS)

    assert os.path.exists(
        seg_model_path
    ), "Segmentation model checkpoint was not saved."
    print(f"Segmentation model saved to: {seg_model_path}")

    # 4.2 Train Classification Model
    print("Training Classifier (1 Epoch)...")
    cls_model_path = train_classifier(debug=True, epochs=Config.CLS_EPOCHS)

    assert os.path.exists(
        cls_model_path
    ), "Classification model checkpoint was not saved."
    print(f"Classification model saved to: {cls_model_path}")

    # -------------------------------------------------------------------------
    # 5. Inference Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating Inference Pipeline ===")

    # 5.1 Create a Mini Test Set for Speed
    # We take the first 3 rows from the actual test metadata and save to a temp file
    full_test_df = pd.read_csv(Config.TEST_CSV)
    mini_test_df = full_test_df.head(3).copy()

    mini_test_csv_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")
    mini_test_df.to_csv(mini_test_csv_path, index=False)
    print(
        f"Created mini test set with {len(mini_test_df)} images at {mini_test_csv_path}"
    )

    # 5.2 Run Submission Generation
    # This function loads the models we just trained (saved in Config.CACHE_DIR)
    # and processes the images in the CSV provided.
    submission_output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    generate_submission(
        test_csv_path=mini_test_csv_path, submission_path=submission_output_path
    )

    # 5.3 Verify Submission Output
    assert os.path.exists(submission_output_path), "Submission file was not created."

    submission_df = pd.read_csv(submission_output_path)
    print("\nGenerated Submission Head:")
    print(submission_df.head())

    assert (
        len(submission_df) == 3
    ), "Submission should contain exactly 3 rows matching the mini test set."
    assert (
        "image_id" in submission_df.columns and "labels" in submission_df.columns
    ), "Submission columns are incorrect."

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
