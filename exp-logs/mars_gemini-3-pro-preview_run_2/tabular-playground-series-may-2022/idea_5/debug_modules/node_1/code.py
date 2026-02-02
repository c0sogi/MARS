import os
import torch
import numpy as np
import pandas as pd
from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import ResGLUNet
from library.trainer import Trainer


def run_demonstration():
    print("=== Starting Library Demonstration ===")

    # 1. Setup and Reproducibility
    print("\n[Step 1] Setting random seeds and configuring for speed...")
    set_seed(42)

    # Override Config for rapid demonstration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128  # Smaller batch size for the small debug dataset
    Config.EARLY_STOPPING_PATIENCE = 1
    Config.HIDDEN_DIM = 64  # Reduce model size for speed
    Config.EMBED_DIM = 8
    Config.NUM_BLOCKS = 2

    # Define debug sample size
    DEBUG_SAMPLES = 1000
    print(
        f"Configuration overridden: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Debug Samples={DEBUG_SAMPLES}"
    )

    # 2. Data Loading
    print("\n[Step 2] Loading DataLoaders...")
    # We use load_cached_data=True to use existing cache if available, but debug_samples will slice it
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, debug_samples=DEBUG_SAMPLES
    )

    # Verify DataLoaders
    print("Verifying Train Loader batch structure...")
    first_batch = next(iter(train_loader))

    # Check keys
    assert "cat" in first_batch, "Batch missing 'cat' key"
    assert "cont" in first_batch, "Batch missing 'cont' key"
    assert "target" in first_batch, "Batch missing 'target' key"

    # Check shapes
    cat_shape = first_batch["cat"].shape
    cont_shape = first_batch["cont"].shape
    target_shape = first_batch["target"].shape

    print(
        f"Batch Shapes -> Cat: {cat_shape}, Cont: {cont_shape}, Target: {target_shape}"
    )

    assert cat_shape[1] == 10, f"Expected 10 categorical tokens, got {cat_shape[1]}"
    assert cont_shape[1] == 30, f"Expected 30 continuous features, got {cont_shape[1]}"
    assert target_shape[1] == 1, f"Expected target dim 1, got {target_shape[1]}"

    print("Data Loading verification passed.")

    # 3. Model Initialization
    print("\n[Step 3] Initializing ResGLUNet model...")
    device = torch.device(Config.DEVICE)
    model = ResGLUNet().to(device)

    print("Verifying Model Forward Pass...")
    # Move batch to device
    x_cat = first_batch["cat"].to(device)
    x_cont = first_batch["cont"].to(device)

    with torch.no_grad():
        output = model(x_cat, x_cont)

    print(f"Model Output Shape: {output.shape}")
    assert (
        output.shape == target_shape
    ), f"Output shape mismatch. Expected {target_shape}, got {output.shape}"
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Model output (probabilities) out of range [0, 1]"

    print("Model verification passed.")

    # 4. Training
    print("\n[Step 4] Starting Training Loop (Demonstration)...")
    trainer = Trainer(model, device=device)

    # Run fit
    best_auc = trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    print(f"Training finished. Best AUC: {best_auc:.4f}")
    assert 0 <= best_auc <= 1, "AUC score out of valid range"
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"

    print("Training verification passed.")

    # 5. Inference
    print("\n[Step 5] Running Inference on Test Set...")
    predictions = trainer.predict(test_loader)

    print(f"Predictions generated. Shape: {predictions.shape}")

    # Verify prediction count matches test loader dataset size
    # Note: test_loader might drop last if drop_last=True, but here it is False for test/val
    expected_len = len(test_loader.dataset)
    assert (
        len(predictions) == expected_len
    ), f"Prediction count mismatch. Expected {expected_len}, got {len(predictions)}"

    # Verify submission file creation logic
    print("Generating submission file...")
    # We need to slice the metadata to match the debug samples for the IDs to align
    test_meta = pd.read_csv(Config.TEST_METADATA)
    test_ids = test_meta["id"].values[:DEBUG_SAMPLES]

    submission = pd.DataFrame({"id": test_ids, "target": predictions})

    # Save to a demo path to avoid overwriting main experiment if needed,
    # but here we use the Config path as requested by the task structure.
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_sub.shape == (
        DEBUG_SAMPLES,
        2,
    ), f"Submission shape mismatch. Expected ({DEBUG_SAMPLES}, 2)"
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission columns missing"

    print("Inference and Submission verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
