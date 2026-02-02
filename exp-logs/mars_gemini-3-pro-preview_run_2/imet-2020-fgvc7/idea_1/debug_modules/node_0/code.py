import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_micro_f1
from library.dataset import ArtworkDataset, get_transforms
from library.model import ArtworkClassifier
from library.train import fit
from library.inference import predict_and_submit


def main():
    print("=== Starting Demonstration Script ===")

    # --- 1. Configuration Setup ---
    # We override some Config parameters to ensure the demo runs quickly and
    # writes to a specific demo directory.
    print("\n[1] Configuring environment...")

    # Set paths for this specific run
    Config.IDEA_DIR = "./working/demo_run"
    Config.MODEL_PATH = os.path.join(Config.IDEA_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.IDEA_DIR, "demo_submission.csv")

    # Setup directories and seeds
    Config.setup()
    set_seed(Config.seed)

    print(f"    Working Directory: {Config.IDEA_DIR}")
    print(f"    Model Path: {Config.MODEL_PATH}")

    # --- 2. Dataset Verification ---
    print("\n[2] Verifying Dataset Logic...")

    # Instantiate the training dataset
    # We use the pre-generated metadata files located in ./metadata
    train_ds = ArtworkDataset(
        metadata_path=Config.TRAIN_METADATA,
        input_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="train", image_size=Config.image_size),
        mode="train",
        num_classes=Config.num_classes,
    )

    # Basic assertions
    assert len(train_ds) > 0, "Training dataset should not be empty."
    print(f"    Training dataset size: {len(train_ds)}")

    # Fetch a single sample
    img, target = train_ds[0]

    # Verify Image Tensor
    # Shape should be (3, H, W)
    assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
    assert img.shape == (
        3,
        Config.image_size,
        Config.image_size,
    ), f"Expected image shape (3, {Config.image_size}, {Config.image_size}), got {img.shape}"

    # Verify Target Tensor
    # Shape should be (Num_Classes,)
    assert isinstance(target, torch.Tensor), "Target should be a torch.Tensor"
    assert target.shape == (
        Config.num_classes,
    ), f"Expected target shape ({Config.num_classes},), got {target.shape}"
    assert target.dtype == torch.float32, "Target dtype should be float32"

    # Check if target is binary (multi-hot)
    unique_vals = torch.unique(target)
    for v in unique_vals:
        assert v.item() in [0.0, 1.0], f"Target values must be 0 or 1, found {v.item()}"

    print("    Dataset verification passed.")

    # --- 3. Model Verification ---
    print("\n[3] Verifying Model Logic...")

    device = Config.device
    model = ArtworkClassifier(
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        pretrained=False,  # False for speed in initialization, though fit() uses True
    )
    model.to(device)
    model.eval()

    # Create a dummy batch
    dummy_input = torch.randn(2, 3, Config.image_size, Config.image_size).to(device)

    with torch.no_grad():
        logits = model(dummy_input)

    # Verify Output Shape: (Batch_Size, Num_Classes)
    assert logits.shape == (
        2,
        Config.num_classes,
    ), f"Expected output shape (2, {Config.num_classes}), got {logits.shape}"

    print("    Model verification passed.")

    # --- 4. Training Demonstration ---
    print("\n[4] Running Training Loop (Debug Mode)...")

    # We run fit() with debug=True.
    # This subsets the data to 100 samples and runs quickly.
    # We also reduce epochs to 1 for speed.
    try:
        fit(
            epochs=1,
            batch_size=16,
            learning_rate=1e-3,
            debug=True,  # Crucial for speed
            num_workers=2,  # Reduce workers for small debug run
            patience=1,
        )
    except Exception as e:
        print(f"    Training failed with error: {e}")
        raise e

    # Verify model artifact creation
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file was not created at {Config.MODEL_PATH}")

    print("    Training complete. Model saved.")

    # --- 5. Inference Demonstration ---
    print("\n[5] Running Inference...")

    # Run prediction using the model we just trained
    try:
        predict_and_submit(
            model_path=Config.MODEL_PATH,
            metadata_path=Config.TEST_METADATA,
            output_path=Config.SUBMISSION_PATH,
            batch_size=16,
            num_workers=2,
            device=Config.device,
            threshold=0.5,
        )
    except Exception as e:
        print(f"    Inference failed with error: {e}")
        raise e

    # Verify submission file creation
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    print("    Inference complete. Submission saved.")

    # --- 6. Submission & Metric Validation ---
    print("\n[6] Validating Output and Metrics...")

    # Load submission
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["id", "attribute_ids"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check content format (attribute_ids should be string or NaN, id should be string)
    assert len(df_sub) > 0, "Submission file is empty."

    # Verify Metric Calculation Logic
    # Case 1: Perfect match
    y_true = np.array([[0, 1, 0], [1, 0, 1]])
    y_pred_logits = np.array(
        [[-10, 10, -10], [10, -10, 10]]
    )  # Sigmoid(10) ~ 1, Sigmoid(-10) ~ 0

    score = calculate_micro_f1(y_pred_logits, y_true, threshold=0.5, from_logits=True)
    assert np.isclose(
        score, 1.0
    ), f"Expected F1 score 1.0 for perfect match, got {score}"

    # Case 2: No overlap
    y_pred_logits_wrong = np.array([[10, -10, 10], [-10, 10, -10]])
    score_wrong = calculate_micro_f1(
        y_pred_logits_wrong, y_true, threshold=0.5, from_logits=True
    )
    assert np.isclose(
        score_wrong, 0.0
    ), f"Expected F1 score 0.0 for no overlap, got {score_wrong}"

    print("    Metric validation passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
