import os
import sys
import torch
import pandas as pd
import numpy as np

# Import components from the provided library
from library.config import Config
from library.utils import seed_everything, get_device, load_model
from library.data import get_loaders, calculate_class_weights
from library.model import AppleResNet34
from library.calibration import run_calibration_phase
from library.production import run_production_phase
from library.engine import generate_submission


def main():
    print(">>> Starting Apple Disease Detection Pipeline Demo")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config parameters for a quick demonstration
    print("Configuring for fast execution...")
    Config.MAX_EPOCHS = 1  # Train for only 1 epoch per fold/seed
    Config.N_FOLDS = 2  # Use only 2 folds for calibration
    Config.SEEDS = [42]  # Use single seed for production ensemble
    Config.BATCH_SIZE = 16  # Reduced batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Set random seeds for reproducibility
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n>>> [1/5] Verifying Data Loading...")

    # Calculate and verify class weights
    # load_cached_data=False forces recalculation to test logic
    weights = calculate_class_weights(load_cached_data=False)
    print(f"Class Weights: {weights}")
    assert isinstance(weights, torch.Tensor), "Weights should be a Tensor"
    assert len(weights) == Config.NUM_CLASSES, f"Expected {Config.NUM_CLASSES} weights"

    # Get Calibration Loaders (Fold 0)
    train_loader, val_loader = get_loaders(
        mode="calibration", fold=0, load_cached_data=False
    )

    # Verify Train Batch
    images, targets = next(iter(train_loader))
    print(f"Train Batch - Images: {images.shape}, Targets: {targets.shape}")

    # Assertions for shapes
    expected_img_shape = (
        Config.BATCH_SIZE,
        Config.CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    expected_target_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Got {images.shape}"
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Got {targets.shape}"

    print("Data loading verified successfully.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n>>> [2/5] Verifying Model Architecture...")

    # Instantiate model
    model = AppleResNet34(pretrained=True)
    model.to(device)
    model.eval()

    # Run a forward pass with the batch fetched earlier
    with torch.no_grad():
        outputs = model(images.to(device))

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == expected_target_shape, "Model output shape mismatch"

    print("Model architecture verified successfully.")

    # ==========================================
    # 4. Calibration Phase (Stage 1)
    # ==========================================
    print("\n>>> [3/5] Running Calibration Phase (Stage 1)...")
    print(f"Running {Config.N_FOLDS} folds for {Config.MAX_EPOCHS} epoch(s)...")

    # Run calibration to find optimal epoch
    optimal_epoch = run_calibration_phase(
        max_epochs=Config.MAX_EPOCHS, load_cached_data=False
    )

    print(f"Optimal Epoch Determined: {optimal_epoch}")
    assert isinstance(optimal_epoch, int), "Optimal epoch must be an integer"
    assert optimal_epoch > 0, "Optimal epoch must be positive"

    print("Calibration phase completed.")

    # ==========================================
    # 5. Production Phase (Stage 2)
    # ==========================================
    print("\n>>> [4/5] Running Production Phase (Stage 2)...")
    print(
        f"Training on full dataset for {optimal_epoch} epoch(s) using seeds: {Config.SEEDS}"
    )

    # Train final model(s)
    model_paths = run_production_phase(
        optimal_epoch=optimal_epoch, load_cached_data=False
    )

    print(f"Models saved at: {model_paths}")
    assert len(model_paths) == len(
        Config.SEEDS
    ), "Number of saved models matches number of seeds"
    for path in model_paths:
        assert os.path.exists(path), f"Model file not found at {path}"

    print("Production phase completed.")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n>>> [5/5] Generating Submission...")

    # Load the trained model (using the first seed for this demo)
    best_model_path = model_paths[0]
    inference_model = AppleResNet34(pretrained=False)  # Architecture only
    inference_model = load_model(inference_model, best_model_path, device)

    # Get Test Loader
    test_loader = get_loaders(mode="test")

    # Define output path
    submission_file = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Generate predictions
    generate_submission(inference_model, test_loader, device, submission_file)

    # Verify Submission
    assert os.path.exists(submission_file), "Submission file was not created"

    df_sub = pd.read_csv(submission_file)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"First 5 rows:\n{df_sub.head()}")

    # Check constraints
    # Test set has 183 images
    assert len(df_sub) == 183, f"Expected 183 rows, got {len(df_sub)}"

    # Check columns
    expected_cols = ["image_id"] + Config.TARGET_COLS
    assert sorted(df_sub.columns.tolist()) == sorted(
        expected_cols
    ), "Column mismatch in submission"

    # Check probability validity
    probs = df_sub[Config.TARGET_COLS].values
    assert (probs >= 0).all() and (
        probs <= 1.0 + 1e-5
    ).all(), "Probabilities out of range [0, 1]"

    print("Submission generated and verified successfully.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
