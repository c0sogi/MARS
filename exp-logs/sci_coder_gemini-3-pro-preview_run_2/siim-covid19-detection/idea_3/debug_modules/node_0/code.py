import os
import sys
import torch
import pandas as pd
import numpy as np

# =============================================================================
# 1. Configuration Patching
# =============================================================================
# We modify the Config class before other modules use it to optimize for the
# demonstration environment (speed and memory).
from library.config import Config

Config.NUM_EPOCHS = 1
Config.BATCH_SIZE = 2
Config.IMG_SIZE = 512  # Reduced from 1024 to ensure speed on demo run
Config.WORKING_DIR = "./working/demo_execution"
Config.SUBMISSION_DIR = "./working/demo_execution"
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# 2. Import Library Modules
# =============================================================================
from library import utils
from library.dataset import CovidDataset, get_transforms
from library.model import CovidMultiTaskModel
from library import engine


def run_demo():
    # Set seeds for reproducibility
    utils.seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running demo on device: {device}")

    # =========================================================================
    # 3. Dataset and DataLoader Demonstration
    # =========================================================================
    print("\n=== Dataset Demonstration ===")

    # Initialize dataset (Train subset)
    train_dataset = CovidDataset(
        subset="train", transforms=get_transforms("train"), load_cached_data=True
    )

    print(f"Total dataset length: {len(train_dataset)}")

    # Fetch a single sample to verify structure
    # We iterate briefly to ensure we get a valid image (in case of read errors)
    sample_img, sample_target, sample_id = None, None, None
    for i in range(len(train_dataset)):
        s_img, s_target, s_id = train_dataset[i]
        if s_img is not None:
            sample_img, sample_target, sample_id = s_img, s_target, s_id
            break

    # Assertions to verify data integrity
    assert sample_img is not None, "Failed to load any images from dataset."
    assert isinstance(sample_img, torch.Tensor), "Image should be a Tensor."
    assert sample_img.ndim == 3, f"Image should be (C, H, W), got {sample_img.shape}"
    # Check dimensions match Config.IMG_SIZE (Albumentations resize)
    assert sample_img.shape[1] == Config.IMG_SIZE, "Image height mismatch."
    assert sample_img.shape[2] == Config.IMG_SIZE, "Image width mismatch."

    assert "boxes" in sample_target, "Target should contain 'boxes'."
    assert "labels" in sample_target, "Target should contain 'labels'."
    assert "study_label" in sample_target, "Target should contain 'study_label'."

    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Target Keys: {list(sample_target.keys())}")
    print(f"Sample Study Label: {sample_target['study_label']}")

    # Test Collate Function with a small batch
    batch_list = [train_dataset[i] for i in range(2) if train_dataset[i][0] is not None]
    if len(batch_list) > 0:
        batched_imgs, batched_targets, batched_ids = utils.collate_fn(batch_list)
        assert batched_imgs.shape[0] == len(batch_list), "Batch size mismatch."
        print(f"Collate function successful. Batch shape: {batched_imgs.shape}")

    # =========================================================================
    # 4. Model Demonstration
    # =========================================================================
    print("\n=== Model Demonstration ===")
    model = CovidMultiTaskModel()
    model.to(device)

    # Prepare batch for model
    b_imgs = batched_imgs.to(device)
    b_targets = [{k: v.to(device) for k, v in t.items()} for t in batched_targets]

    # Test Training Forward Pass
    print("Testing Training Forward Pass...")
    model.train()
    loss_dict = model(b_imgs, b_targets)

    # Verify Loss Dictionary
    expected_losses = [
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_study",
    ]
    for loss_name in expected_losses:
        assert loss_name in loss_dict, f"Missing {loss_name} in output."
        assert not torch.isnan(loss_dict[loss_name]), f"{loss_name} is NaN."

    print("  Losses computed successfully:")
    for k, v in loss_dict.items():
        print(f"    {k}: {v.item():.4f}")

    # Test Inference Forward Pass
    print("Testing Inference Forward Pass...")
    model.eval()
    with torch.no_grad():
        detections = model(b_imgs)

    assert len(detections) == len(b_imgs), "Output length matches batch size."
    assert "boxes" in detections[0], "Detections should contain boxes."
    assert (
        "study_prediction" in detections[0]
    ), "Detections should contain study_prediction."

    print("  Inference successful.")
    print(f"  Study Prediction Shape: {detections[0]['study_prediction'].shape}")
    print(f"  Detected Boxes Count: {detections[0]['boxes'].shape[0]}")

    # =========================================================================
    # 5. Full Engine Execution (Fit)
    # =========================================================================
    print("\n=== Engine Execution (Fit) ===")
    print("Running engine.fit(debug=True) to simulate full pipeline...")
    print(
        "This will train for 1 epoch on 100 images, validate, and generate submission."
    )

    # engine.fit handles the entire lifecycle:
    # 1. Creates DataLoaders (subsetting data due to debug=True)
    # 2. Initializes Model, Optimizer, Scheduler
    # 3. Runs Training Loop
    # 4. Runs Validation Loop
    # 5. Saves Checkpoints
    # 6. Runs Inference on Test Set
    # 7. Generates Submission CSV
    engine.fit(debug=True)

    # =========================================================================
    # 6. Output Validation
    # =========================================================================
    print("\n=== Validating Outputs ===")

    # Check for Checkpoints
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        # If validation didn't improve (unlikely but possible), check for 'checkpoint.pth'
        checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoint.pth")

    assert os.path.exists(checkpoint_path), "Model checkpoint was not created."
    print(f"Checkpoint found at: {checkpoint_path}")

    # Check for Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Verify Submission Schema
    assert "id" in submission_df.columns, "Submission missing 'id' column."
    assert (
        "PredictionString" in submission_df.columns
    ), "Submission missing 'PredictionString' column."
    assert len(submission_df) > 0, "Submission file is empty."

    # Verify content format (simple check on first row)
    first_pred = submission_df.iloc[0]["PredictionString"]
    assert isinstance(first_pred, str), "PredictionString should be a string."

    print("Submission file generated and verified successfully.")
    print(submission_df.head())

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
