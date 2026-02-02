import os
import torch
import numpy as np
import pandas as pd
import warnings
from torch.optim.swa_utils import AveragedModel

# Import provided library modules
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import IcebergResNet18
from library.training import Trainer
from library.inference import (
    validate_model,
    predict_with_tta,
    load_test_ids,
    save_submission,
)

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def demonstrate_pipeline():
    print("=== Starting Demonstration Pipeline ===")

    # 1. Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_execution"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Set seed for reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading & Verification
    print("\n--- Step 1: Verifying Data Loading ---")
    # We use debug=True to load only 100 samples for speed
    train_loader, val_loader, test_loader = get_dataloaders(
        input_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        cache_dir=CACHE_DIR,
        batch_size=8,
        num_workers=0,  # Avoid multiprocessing overhead for demo
        load_cached_data=False,  # Force processing logic to run
        debug=True,
    )

    # Fetch a batch to verify shapes
    images, angles, labels = next(iter(train_loader))

    # Assertions
    # Image shape: (Batch, 3, 224, 224) - 3 channels because of composite creation in library.data
    assert images.shape == (8, 3, 224, 224), f"Unexpected image shape: {images.shape}"
    # Angle shape: (Batch,)
    assert angles.shape == (8,), f"Unexpected angle shape: {angles.shape}"
    # Label shape: (Batch,)
    assert labels.shape == (8,), f"Unexpected label shape: {labels.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Verification
    print("\n--- Step 2: Verifying Model Architecture ---")
    model = IcebergResNet18(dropout_rate=0.5).to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(images, angles)

    # Assert output shape (Batch, 1) - Logits
    assert outputs.shape == (8, 1), f"Unexpected output shape: {outputs.shape}"
    print("Model forward pass verified successfully.")

    # 4. Training Pipeline (Fast-Forward)
    print("\n--- Step 3: Running Training Pipeline (Debug Mode) ---")
    trainer = Trainer(
        input_dir=INPUT_DIR, metadata_dir=METADATA_DIR, working_dir=WORKING_DIR
    )

    # Phase 1: Calibration
    # We use very few epochs to ensure this runs quickly
    print("Running Calibration Phase...")
    optimal_steps = trainer.run_calibration_phase(
        batch_size=16, max_epochs=2, debug=True  # Minimal epochs for demo
    )
    assert isinstance(optimal_steps, int) and optimal_steps > 0
    print(f"Optimal steps determined: {optimal_steps}")

    # Phase 2: Production
    # Train a single model for demonstration
    print("Running Production Phase...")
    saved_models = trainer.run_production_phase(
        optimal_steps=optimal_steps,
        batch_size=16,
        num_models=1,  # Single model for speed
        debug=True,
    )

    assert len(saved_models) > 0, "No models were saved."
    assert os.path.exists(
        saved_models[0]
    ), f"Saved model file not found: {saved_models[0]}"
    print(f"Model saved at: {saved_models[0]}")

    # 5. Inference & Submission
    print("\n--- Step 4: Inference and Submission ---")

    # Load the trained SWA model
    # Note: The saved state_dict is from an AveragedModel wrapper
    base_model = IcebergResNet18(dropout_rate=0.5).to(device)
    swa_model = AveragedModel(base_model).to(device)

    checkpoint_path = saved_models[0]
    swa_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Model loaded successfully.")

    # Validate on Validation Set (Logic Check)
    # Note: val_loader from debug mode has labels
    val_loss, val_preds, val_targets = validate_model(swa_model, val_loader, device)
    print(f"Validation Log Loss (Debug Set): {val_loss:.4f}")

    # Predict on Test Set
    # Note: test_loader from debug mode has 100 samples, but we need full test set for submission structure check.
    # We will reload the test loader with debug=False to get the full ID list structure,
    # but we will limit the inference loop if needed or just run it (321 samples is small).

    print("Reloading full test set for submission generation...")
    _, _, full_test_loader = get_dataloaders(
        input_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        cache_dir=CACHE_DIR,
        batch_size=32,
        num_workers=0,
        load_cached_data=True,  # Use cache if available from previous steps
        debug=False,  # We want full test set IDs
    )

    print("Generating predictions with TTA...")
    test_preds = predict_with_tta(swa_model, full_test_loader, device)

    # Load Test IDs
    test_ids = load_test_ids(METADATA_DIR)

    # Verify alignment
    assert len(test_preds) == len(
        test_ids
    ), f"Mismatch: {len(test_preds)} predictions vs {len(test_ids)} IDs"

    # Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    save_submission(test_preds, test_ids, submission_path)

    # Verify File
    assert os.path.exists(submission_path)
    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")
    print(df_sub.head())

    print("\n=== Pipeline Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_pipeline()
