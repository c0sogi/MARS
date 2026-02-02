import os
import torch
import pandas as pd
import numpy as np
import sys

# Import library components
from library.config import Config, seed_everything
from library.dataset import get_data_loaders, get_test_loader
from library.model import AppleResNet34
from library.loss import WeightedSoftCrossEntropy, get_class_weights
from library.sam import SAM
from library.train_eval import train_single_fold, generate_submission


def main():
    print("==== Starting Demonstration of Apple Disease Detection Pipeline ====")

    # 1. Setup and Configuration
    # Enable Debug mode to reduce epochs (2), splits (2), and dataset size (subsampled)
    Config.setup(debug=True)
    seed_everything(42)

    # Ensure working directories exist (Config.setup does this, but double check logic)
    assert os.path.exists(Config.WORKING_DIR), "Working directory not created."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print("\n[Demo] Data Loading...")
    # Use a specific split seed to trigger the stratified shuffle-split logic
    train_loader, val_loader = get_data_loaders(split_seed=Config.SEEDS[0])

    # Fetch one batch
    images, targets = next(iter(train_loader))

    # Verify Shapes
    # Batch size might be smaller if dataset is small and drop_last=True/False interaction occurs,
    # but in debug mode with 64 samples and batch size 32, we expect 32.
    print(f"  Image Batch Shape: {images.shape}")
    print(f"  Target Batch Shape: {targets.shape}")

    assert len(images.shape) == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert (
        images.shape[2] == Config.IMAGE_SIZE and images.shape[3] == Config.IMAGE_SIZE
    ), f"Images should be resized to {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}"
    assert (
        targets.shape[1] == Config.NUM_CLASSES
    ), f"Targets should have {Config.NUM_CLASSES} classes"

    print("  -> Data Loading Verified.")

    # ==========================================
    # 3. Model Demonstration
    # ==========================================
    print("\n[Demo] Model Initialization & Forward Pass...")
    model = AppleResNet34(pretrained=True).to(device)

    # Verify Weight Initialization Logic
    model.check_initial_weights()

    # Forward Pass
    images = images.to(device)
    logits = model(images)

    print(f"  Logits Shape: {logits.shape}")
    assert logits.shape == (
        images.size(0),
        Config.NUM_CLASSES,
    ), "Output logits shape mismatch."

    print("  -> Model Verified.")

    # ==========================================
    # 4. Loss Function Demonstration
    # ==========================================
    print("\n[Demo] Loss Function...")
    # Load metadata to calculate weights
    df_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    class_weights = get_class_weights(df_meta, Config.CLASS_NAMES, device=device)

    print(f"  Class Weights: {class_weights.cpu().numpy()}")

    criterion = WeightedSoftCrossEntropy(class_weights=class_weights)
    targets = targets.to(device)

    loss = criterion(logits, targets)
    print(f"  Calculated Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("  -> Loss Function Verified.")

    # ==========================================
    # 5. Optimizer (SAM) Demonstration
    # ==========================================
    print("\n[Demo] SAM Optimizer...")
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=1e-3)

    # Define closure for SAM
    def closure():
        optimizer.zero_grad()
        out = model(images)
        l = criterion(out, targets)
        l.backward()
        return l

    # Check parameters before step
    param_before = next(model.parameters()).clone()

    # Perform step
    loss_val = optimizer.step(closure)

    # Check parameters after step
    param_after = next(model.parameters())

    assert not torch.equal(
        param_before, param_after
    ), "Optimizer did not update parameters."
    print(f"  Optimization Step Loss: {loss_val.item():.6f}")
    print("  -> SAM Optimizer Verified.")

    # ==========================================
    # 6. Full Training Pipeline Integration
    # ==========================================
    print("\n[Demo] Running Training Pipeline (Single Fold)...")
    # We run training for the first seed in the list.
    # Config.DEBUG is True, so this runs for 2 epochs on a small subset.
    seed_to_train = Config.SEEDS[0]

    best_auc = train_single_fold(split_seed=seed_to_train, fold_idx=0)

    # Verify Model Artifacts
    expected_model_path = os.path.join(
        Config.MODEL_DIR, f"resnet34_seed_{seed_to_train}.pth"
    )
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found at {expected_model_path}"
    print(f"  -> Training completed. Model saved to {expected_model_path}")

    # ==========================================
    # 7. Inference & Submission Integration
    # ==========================================
    print("\n[Demo] Generating Submission...")

    # generate_submission iterates over Config.SEEDS.
    # In Debug mode, Config.SEEDS has 2 seeds. We only trained one above.
    # The function handles missing models by skipping them, so this is safe.
    generate_submission()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission Shape: {df_sub.shape}")

    # Load test metadata to check expected length
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    expected_rows = len(df_test)

    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    expected_cols = ["image_id"] + Config.CLASS_NAMES
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check values are probabilities
    probs = df_sub[Config.CLASS_NAMES].values
    assert (probs >= 0).all() and (
        probs <= 1.00001
    ).all(), "Probabilities out of range [0, 1]"

    print("  -> Submission Verified.")

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
