import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_qwk, get_llrd_optimizer_params
from library.dataset import get_dataloaders
from library.model import EssayScorerModel
from library.trainer import train_fn, inference_fn


def run_demo():
    # =========================================================================
    # 1. Configuration Setup for Fast Demo
    # =========================================================================
    print("--- 1. Setting up Demo Configuration ---")

    class DemoConfig(Config):
        # Enable debug to use data subsets (100 train, 50 val, 50 test)
        debug = True

        # Use a tiny model for speed verification
        model_name = "prajjwal1/bert-tiny"

        # Reduce sequence length for speed
        max_length = 64

        # Training settings for quick turnaround
        epochs = 1
        train_batch_size = 4
        valid_batch_size = 4
        gradient_accumulation_steps = 1
        val_check_interval = 1.0  # Validate only at the end of epoch

        # Output paths
        working_dir = "./working/demo_execution"
        os.makedirs(working_dir, exist_ok=True)

        train_cache_path = os.path.join(working_dir, "train_processed.parquet")
        val_cache_path = os.path.join(working_dir, "val_processed.parquet")
        test_cache_path = os.path.join(working_dir, "test_processed.parquet")
        model_save_path = os.path.join(working_dir, "demo_model.pth")
        submission_path = os.path.join(working_dir, "submission.csv")

        # Ensure device is set correctly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DemoConfig()

    # Clean up previous demo run if exists
    if os.path.exists(cfg.working_dir):
        shutil.rmtree(cfg.working_dir)
    os.makedirs(cfg.working_dir, exist_ok=True)

    # =========================================================================
    # 2. Verify Utilities
    # =========================================================================
    print("--- 2. Verifying Utilities ---")
    set_seed(cfg.seed)

    # Test QWK Metric
    y_true = np.array([1, 2, 3, 4, 5, 6])
    y_pred_perfect = np.array(
        [1.1, 1.9, 3.0, 4.0, 5.1, 5.9]
    )  # Should round to perfect match
    score = compute_qwk(y_true, y_pred_perfect)
    print(f"Computed QWK (Perfect): {score}")
    assert score > 0.99, "QWK calculation logic failed for perfect predictions."

    y_pred_bad = np.array([6, 5, 4, 3, 2, 1])
    score_bad = compute_qwk(y_true, y_pred_bad)
    print(f"Computed QWK (Inverse): {score_bad}")
    assert score_bad < 0.5, "QWK calculation logic failed for bad predictions."

    # =========================================================================
    # 3. Verify Data Pipeline
    # =========================================================================
    print("--- 3. Verifying Data Pipeline ---")
    train_loader, val_loader, test_loader = get_dataloaders(cfg, load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Assert keys
    expected_keys = ["input_ids", "attention_mask", "essay_id", "labels"]
    for k in expected_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Assert shapes
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert (
        input_ids.shape[0] == cfg.train_batch_size
    ), f"Batch size mismatch. Expected {cfg.train_batch_size}, got {input_ids.shape[0]}"
    assert input_ids.shape[1] <= cfg.max_length, f"Sequence length exceeds max_length."
    assert labels.shape[0] == cfg.train_batch_size, "Labels shape mismatch."

    print("Data Pipeline Verified.")

    # =========================================================================
    # 4. Verify Model Architecture
    # =========================================================================
    print("--- 4. Verifying Model Architecture ---")
    model = EssayScorerModel(cfg, pretrained=True)
    model.to(cfg.device)

    # Move batch to device
    b_input_ids = batch["input_ids"].to(cfg.device)
    b_mask = batch["attention_mask"].to(cfg.device)

    # Forward pass
    output = model(b_input_ids, b_mask)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (cfg.train_batch_size, 1), "Model output shape incorrect."
    assert not torch.isnan(output).any(), "Model produced NaN outputs."

    print("Model Architecture Verified.")

    # =========================================================================
    # 5. Verify Optimizer & LLRD
    # =========================================================================
    print("--- 5. Verifying Optimizer Setup ---")
    optimizer_params = get_llrd_optimizer_params(
        model,
        base_lr=cfg.lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
        llrd_decay=cfg.llrd_decay,
    )

    assert isinstance(optimizer_params, list), "Optimizer params should be a list."
    assert len(optimizer_params) > 0, "Optimizer params list is empty."
    assert "params" in optimizer_params[0], "Optimizer groups malformed."

    optimizer = torch.optim.AdamW(optimizer_params, lr=cfg.lr)
    print("Optimizer initialized successfully.")

    # =========================================================================
    # 6. Verify Training Loop (Single Epoch)
    # =========================================================================
    print("--- 6. Verifying Training Loop ---")

    # Setup Scheduler and Scaler
    num_update_steps = len(train_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_update_steps
    )
    scaler = GradScaler()

    # Run Train Function
    # We pass -inf as best_score to ensure it tries to save at least once if val score is valid
    best_score = train_fn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=cfg.device,
        cfg=cfg,
        best_score=-np.inf,
        epoch=0,
    )

    print(f"Training Loop Finished. Best Score: {best_score}")
    assert os.path.exists(cfg.model_save_path), "Model file was not saved."

    # =========================================================================
    # 7. Verify Inference & Submission
    # =========================================================================
    print("--- 7. Verifying Inference & Submission ---")

    # Load the saved model
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))

    ids, raw_preds = inference_fn(model, test_loader, cfg.device)

    assert len(ids) == len(raw_preds), "Mismatch between IDs and predictions count."

    # Process predictions
    final_preds = np.clip(raw_preds, 1, 6)
    final_preds = np.round(final_preds).astype(int)

    # Create DataFrame
    submission_df = pd.DataFrame({"essay_id": ids, "score": final_preds})

    # Check format
    print(submission_df.head())
    assert list(submission_df.columns) == [
        "essay_id",
        "score",
    ], "Submission columns incorrect."
    assert (
        submission_df["score"].min() >= 1 and submission_df["score"].max() <= 6
    ), "Scores out of range."

    # Save
    submission_df.to_csv(cfg.submission_path, index=False)
    assert os.path.exists(cfg.submission_path), "Submission file not created."

    print("--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
