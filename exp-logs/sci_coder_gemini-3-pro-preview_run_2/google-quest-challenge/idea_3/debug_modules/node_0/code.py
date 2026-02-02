import os
import shutil
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, logging as transformers_logging

# Import from the provided library
import library.config
from library.config import Config
from library.dataset import get_dataloaders, QuestDataset
from library.model import CausalAwareSiameseDeberta
from library.metrics import compute_spearmanr
from library.trainer import Trainer


def run_demo():
    # ==========================================
    # 0. Setup & Configuration Override
    # ==========================================
    print(">>> [Step 0] Configuring environment for rapid demonstration...")

    # Suppress verbose warnings
    transformers_logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Override Config class attributes for a fast debug run
    # We modify the class directly so all instances pick up these changes
    library.config.Config.DEBUG = True
    library.config.Config.DEBUG_SAMPLES = 20  # Small subset for speed
    library.config.Config.EPOCHS = 1  # Single epoch
    library.config.Config.TRAIN_BATCH_SIZE = 4
    library.config.Config.VALID_BATCH_SIZE = 4
    library.config.Config.WORKING_DIR = "./working/demo_script_run/"

    # Clean up previous run if exists
    if os.path.exists(library.config.Config.WORKING_DIR):
        shutil.rmtree(library.config.Config.WORKING_DIR)
    os.makedirs(library.config.Config.WORKING_DIR, exist_ok=True)

    cfg = Config()
    print(f"    Debug Mode: {cfg.DEBUG}")
    print(f"    Working Dir: {cfg.WORKING_DIR}")

    # ==========================================
    # 1. Dataset & DataLoader Verification
    # ==========================================
    print("\n>>> [Step 1] Verifying Dataset and Dataloaders...")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    # Generate dataloaders
    # This will process the debug subset of data and cache it
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=False
    )

    # Fetch one batch from train_loader
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = [
        "input_ids_q",
        "attention_mask_q",
        "input_ids_a",
        "attention_mask_a",
        "labels",
    ]
    for key in expected_keys:
        if key not in batch:
            raise AssertionError(f"Missing key '{key}' in batch.")

    # Verify Shapes
    # Batch size should be 4 (as set in config override)
    # Sequence length is 512 (MAX_LEN)
    # Labels should be (4, 30)
    assert batch["input_ids_q"].shape == (
        4,
        512,
    ), f"Incorrect Q input shape: {batch['input_ids_q'].shape}"
    assert batch["input_ids_a"].shape == (
        4,
        512,
    ), f"Incorrect A input shape: {batch['input_ids_a'].shape}"
    assert batch["labels"].shape == (
        4,
        30,
    ), f"Incorrect labels shape: {batch['labels'].shape}"

    print("    Batch shapes verified successfully.")
    print(f"    Input Q Shape: {batch['input_ids_q'].shape}")
    print(f"    Labels Shape: {batch['labels'].shape}")

    # ==========================================
    # 2. Model Architecture Verification
    # ==========================================
    print("\n>>> [Step 2] Verifying Model Architecture...")

    model = CausalAwareSiameseDeberta()
    model.eval()  # Set to eval mode for deterministic behavior

    # Move batch to CPU (since we are just testing logic, GPU not strictly required but handled if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = {
        "input_ids_q": batch["input_ids_q"].to(device),
        "attention_mask_q": batch["attention_mask_q"].to(device),
        "input_ids_a": batch["input_ids_a"].to(device),
        "attention_mask_a": batch["attention_mask_a"].to(device),
    }

    if "token_type_ids_q" in batch:
        inputs["token_type_ids_q"] = batch["token_type_ids_q"].to(device)
    if "token_type_ids_a" in batch:
        inputs["token_type_ids_a"] = batch["token_type_ids_a"].to(device)

    with torch.no_grad():
        logits = model(**inputs)

    # Verify Output Shape
    # Should be (Batch_Size, 30)
    assert logits.shape == (
        4,
        30,
    ), f"Model output shape mismatch. Expected (4, 30), got {logits.shape}"

    print("    Model forward pass successful.")
    print(f"    Logits Shape: {logits.shape}")

    # ==========================================
    # 3. Metric Verification
    # ==========================================
    print("\n>>> [Step 3] Verifying Metric (Spearman Correlation)...")

    # Case A: Perfect Correlation
    y_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y_pred = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    score_perfect = compute_spearmanr(y_true, y_pred)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Expected 1.0 for perfect correlation, got {score_perfect}"

    # Case B: Inverse Correlation
    y_pred_inv = np.array([[0.5, 0.6], [0.3, 0.4], [0.1, 0.2]])
    score_inv = compute_spearmanr(y_true, y_pred_inv)
    assert np.isclose(
        score_inv, -1.0
    ), f"Expected -1.0 for inverse correlation, got {score_inv}"

    print(
        f"    Metric verification passed. Perfect Score: {score_perfect}, Inverse Score: {score_inv}"
    )

    # ==========================================
    # 4. Trainer Integration Test
    # ==========================================
    print("\n>>> [Step 4] Running Trainer Integration Test (Fit & Predict)...")

    trainer = Trainer()

    # 4.1 Fit
    # This runs the training loop for 1 epoch on the debug dataset
    print("    Starting training loop...")
    test_loader_from_trainer = trainer.fit()

    # Verify best model was saved
    best_model_path = os.path.join(cfg.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print("    Training complete. Best model saved.")

    # 4.2 Predict
    print("    Generating predictions...")
    qa_ids, preds = trainer.predict(test_loader_from_trainer)

    # Verify Prediction Shapes
    # Test set in debug mode should also be limited to DEBUG_SAMPLES (20)
    # However, depending on how _process_split handles test set in debug mode:
    # The dataset.py says: if cfg.DEBUG: df = df.iloc[: cfg.DEBUG_SAMPLES]
    # So we expect 20 predictions.

    expected_samples = cfg.DEBUG_SAMPLES
    assert (
        len(qa_ids) == expected_samples
    ), f"Expected {expected_samples} QA IDs, got {len(qa_ids)}"
    assert preds.shape == (
        expected_samples,
        30,
    ), f"Expected preds shape ({expected_samples}, 30), got {preds.shape}"

    print(f"    Predictions generated successfully. Shape: {preds.shape}")

    # 4.3 Submission Generation
    trainer.generate_submission(qa_ids, preds)
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file not found."

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    assert sub_df.shape == (
        expected_samples,
        31,
    ), f"Submission shape mismatch. Expected ({expected_samples}, 31), got {sub_df.shape}"
    print("    Submission file verified.")

    # ==========================================
    # 5. Cleanup
    # ==========================================
    print("\n>>> [Step 5] Cleaning up...")
    if os.path.exists(cfg.WORKING_DIR):
        shutil.rmtree(cfg.WORKING_DIR)
    # We keep the ./submission folder as it might be useful to inspect,
    # but strictly speaking the prompt allows using ./working for temp files.

    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
