import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_pos_weights
from library.dataset import get_data, AppleDataset
from library.model import AppleDiseaseModel
from library.loss import WeightedBCELoss
from library.train import run_training
from library.inference import run_inference, reconstruct_probabilities


def main():
    # 1. Setup and Configuration Override for Speed
    print(">>> Setting up demonstration environment...")
    seed_everything(Config.SEED)

    # Modify Config for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 150  # Use a small subset of data (Cite debug_lesson_1: Ensure sufficient samples for batching)
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.FOLDS = 2  # Use 2 folds instead of 5
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = "./working/demo_run/submission.csv"

    # Use only one model for the demo to save time (ConvNeXt is usually faster/lighter than EffNetV2-L)
    # We keep the configuration dict but ensure the list has only one item
    Config.MODELS = [m for m in Config.MODELS if "convnext" in m["name"]]
    if not Config.MODELS:
        # Fallback if filter fails
        Config.MODELS = [Config.MODELS[0]]

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Selected Model: {Config.MODELS[0]['name']}")

    # 2. Component Verification: Dataset and DataLoader
    print("\n>>> Verifying Dataset and DataLoader...")
    # Load metadata
    train_df = get_data("train")

    # Verify derived targets exist
    assert "target_rust" in train_df.columns, "target_rust column missing"
    assert "target_scab" in train_df.columns, "target_scab column missing"

    # Initialize Dataset
    dataset = AppleDataset(train_df.head(10), img_size=384, mode="train")

    # Fetch one sample
    image, targets = dataset[0]

    # Verify shapes
    # Image should be [3, 384, 384] (Channels, Height, Width)
    assert image.shape == (3, 384, 384), f"Incorrect image shape: {image.shape}"
    # Targets should be [2] (Rust, Scab)
    assert targets.shape == (2,), f"Incorrect target shape: {targets.shape}"

    print(
        "Dataset verification successful. Image shape: [3, 384, 384], Target shape: [2]"
    )

    # 3. Component Verification: Model and Loss
    print("\n>>> Verifying Model and Loss Logic...")
    device = Config.DEVICE

    # Instantiate Model
    model_cfg = Config.MODELS[0]
    model = AppleDiseaseModel(
        model_name=model_cfg["name"],
        pretrained=False,  # False for speed in demo, though usually True
        num_classes=Config.NUM_TARGETS,
        gem_p=model_cfg["gem_p"],
        num_msd=model_cfg["num_msd"],
        msd_dropout=model_cfg["msd_dropout"],
    ).to(device)
    model.eval()

    # Create dummy batch [Batch=2, C=3, H=384, W=384]
    dummy_input = torch.randn(2, 3, 384, 384).to(device)
    dummy_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(
        device
    )  # [Rust=1, Scab=0], [Rust=0, Scab=1]

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)

    # Verify output shape [Batch, Num_Classes]
    assert logits.shape == (
        2,
        2,
    ), f"Model output shape mismatch. Expected (2, 2), got {logits.shape}"

    # Verify Loss Calculation
    pos_weights = calculate_pos_weights(train_df).to(device)
    criterion = WeightedBCELoss(pos_weights=pos_weights, smoothing=0.0)

    # Compute loss (requires grad usually, but we just check computation)
    # We need to enable grad on logits to check backward if we wanted, but here just forward
    logits.requires_grad = True
    loss = criterion(logits, dummy_targets)

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"

    print("Model and Loss verification successful.")

    # 4. Component Verification: Probability Reconstruction Logic
    print("\n>>> Verifying Probability Reconstruction Logic...")
    # Scenario: High Rust (0.9), Low Scab (0.1)
    # Expected:
    #   Healthy = (1-0.9)*(1-0.1) = 0.1*0.9 = 0.09
    #   Multiple = 0.9*0.1 = 0.09
    #   Rust = 0.9*(1-0.1) = 0.81
    #   Scab = (1-0.9)*0.1 = 0.01
    #   Sum = 1.0

    scores = np.array([[0.9, 0.1]])
    probs = reconstruct_probabilities(scores)

    # Columns: [healthy, multiple_diseases, rust, scab]
    expected = np.array([0.09, 0.09, 0.81, 0.01])

    # Check closeness (allowing for small floating point diffs and normalization in function)
    assert np.allclose(
        probs, expected.reshape(1, 4), atol=1e-4
    ), f"Reconstruction logic failed. Got {probs}, Expected {expected}"

    print("Probability reconstruction logic verified.")

    # 5. Integration: Run Training Pipeline
    print("\n>>> Executing Training Pipeline (Fast Demo)...")
    # This will train for 1 epoch on 2 folds using a small subset
    try:
        run_training()
        print("Training pipeline executed successfully.")
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        raise e

    # Verify checkpoints were created
    # We expect 'best_model_..._fold_0.pth' and 'best_model_..._fold_1.pth'
    safe_model_name = model_cfg["name"].replace(".", "_")
    fold_0_path = os.path.join(
        Config.WORKING_DIR, f"best_model_{safe_model_name}_fold_0.pth"
    )

    if os.path.exists(fold_0_path):
        print(f"Checkpoint verified: {fold_0_path}")
    else:
        raise FileNotFoundError(f"Expected checkpoint not found at {fold_0_path}")

    # 6. Integration: Run Inference Pipeline
    print("\n>>> Executing Inference Pipeline...")
    try:
        run_inference()
        print("Inference pipeline executed successfully.")
    except Exception as e:
        print(f"Inference pipeline failed: {e}")
        raise e

    # 7. Final Submission Verification
    print("\n>>> Verifying Submission File...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    assert list(sub_df.columns) == expected_cols, f"Invalid columns: {sub_df.columns}"

    # Check values are probabilities
    numeric_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    assert (sub_df[numeric_cols].values >= 0).all(), "Negative probabilities found"
    assert (sub_df[numeric_cols].values <= 1.0001).all(), "Probabilities > 1 found"

    print("Submission file verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
