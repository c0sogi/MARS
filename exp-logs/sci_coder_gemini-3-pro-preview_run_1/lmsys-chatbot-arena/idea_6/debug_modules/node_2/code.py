import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, get_device, compute_metrics
from library.data_processing import get_dataloaders
from library.model import SiameseDebertaCrossAttn
from library.trainer import Trainer


def run_demo():
    print("----------------------------------------------------------------")
    print("Starting Library Demonstration Script")
    print("----------------------------------------------------------------")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring Environment for Demo...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed
    Config.NUM_EPOCHS = 1
    # Reduced batch sizes to prevent OOM on 16GB GPU
    Config.TRAIN_BATCH_SIZE = 2
    Config.VALID_BATCH_SIZE = 4
    Config.ACCUMULATION_STEPS = 2

    # Use a specific directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Setup environment (creates directories)
    Config.setup_environment()

    # Set seeds
    seed_everything(Config.SEED)

    device = get_device()
    print(f"Device selected: {device}")

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n[2] Verifying Utility Functions...")

    # Test compute_metrics with dummy data
    y_true_dummy = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred_dummy = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    score = compute_metrics(y_true_dummy, y_pred_dummy)

    print(f"Dummy Log Loss: {score:.4f}")
    assert isinstance(score, float), "compute_metrics should return a float"
    assert score > 0, "Log loss should be positive"

    # ==========================================
    # 3. Data Processing & Loading
    # ==========================================
    print("\n[3] Initializing Data Loaders...")

    # Force reload of cache to ensure processing logic runs
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
        os.makedirs(Config.CACHE_DIR)

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=False  # Force processing
    )

    # Verify Train Loader Batch Structure
    print("Verifying Train Loader batch structure...")
    batch = next(iter(train_loader))

    # Expected keys
    expected_keys = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "scalar_features",
        "labels",
    ]
    for key in expected_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Check dimensions
    # Batch size might be smaller if drop_last=True and dataset size < batch size,
    # but here dataset=50, batch=4, so full batch expected.
    curr_batch_size = batch["labels"].size(0)
    assert (
        curr_batch_size == Config.TRAIN_BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.TRAIN_BATCH_SIZE}, got {curr_batch_size}"

    # Check scalar features shape (Batch, 6)
    assert batch["scalar_features"].shape == (
        curr_batch_size,
        6,
    ), f"Scalar features shape mismatch: {batch['scalar_features'].shape}"

    # Check labels shape (Batch, 3)
    assert batch["labels"].shape == (
        curr_batch_size,
        3,
    ), f"Labels shape mismatch: {batch['labels'].shape}"

    print("Data Loader verification passed.")

    # ==========================================
    # 4. Model Initialization & Forward Pass
    # ==========================================
    print("\n[4] Initializing Model...")

    model = SiameseDebertaCrossAttn()
    model.to(device)

    print("Running forward pass check...")
    # Move batch to device
    inputs = {
        "input_ids_a": batch["input_ids_a"].to(device),
        "attention_mask_a": batch["attention_mask_a"].to(device),
        "input_ids_b": batch["input_ids_b"].to(device),
        "attention_mask_b": batch["attention_mask_b"].to(device),
        "scalar_features": batch["scalar_features"].to(device),
    }

    # Forward pass
    model.train()
    logits = model(**inputs)

    # Verify output shape (Batch, NumClasses=3)
    assert logits.shape == (
        curr_batch_size,
        3,
    ), f"Model output shape mismatch: {logits.shape}"
    print(f"Forward pass successful. Logits shape: {logits.shape}")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("\n[5] Starting Training Loop (Demo)...")

    trainer = Trainer(model, train_loader, val_loader)

    # Run fit (1 epoch as per config override)
    trainer.fit()

    # Verify model was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # ==========================================
    # 6. Inference & Submission Generation
    # ==========================================
    print("\n[6] Running Inference on Test Set...")

    # Load best model
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    test_preds = []

    with torch.no_grad():
        for i, test_batch in enumerate(test_loader):
            inputs_test = {
                "input_ids_a": test_batch["input_ids_a"].to(device),
                "attention_mask_a": test_batch["attention_mask_a"].to(device),
                "input_ids_b": test_batch["input_ids_b"].to(device),
                "attention_mask_b": test_batch["attention_mask_b"].to(device),
                "scalar_features": test_batch["scalar_features"].to(device),
            }

            logits = model(**inputs_test)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            test_preds.append(probs)

            # Limit inference for demo speed
            if i >= 2:
                break

    test_preds = np.concatenate(test_preds, axis=0)
    print(f"Inference complete. Predictions shape (partial): {test_preds.shape}")

    # Generate dummy submission file based on predictions
    # We need IDs from the test metadata.
    # Since we subsampled or broke early, we just take the first N ids.
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    if Config.DEBUG:
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Align lengths
    num_preds = len(test_preds)
    submission_ids = df_test["id"].iloc[:num_preds].values

    submission_df = pd.DataFrame(
        {
            "id": submission_ids,
            "winner_model_a": test_preds[:, 0],
            "winner_model_b": test_preds[:, 1],
            "winner_tie": test_preds[:, 2],
        }
    )

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission file generated at {Config.SUBMISSION_PATH}")

    # Verify submission format
    assert submission_df.shape[1] == 4, "Submission must have 4 columns"
    assert "id" in submission_df.columns, "Submission must have 'id' column"

    print("\n----------------------------------------------------------------")
    print("Demo execution completed successfully.")
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    run_demo()
