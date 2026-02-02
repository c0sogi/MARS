import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import load_and_preprocess, EssayDataset
from library.model import EssayModel
from library.awp import AWP
from library.trainer import train_backbone
from library.stacking import train_stacking, inference

# Configure logging to suppress non-essential output for this demo
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)


def run_demo():
    print("=== Starting Essay Scoring Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Patch Config for speed and demonstration purposes
    print("Configuring environment...")
    Config.working_dir = "./working/demo_run"
    Config.output_dir = os.path.join(Config.working_dir, "output")
    Config.model_dir = os.path.join(Config.working_dir, "checkpoints")
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.submission_dir = os.path.join(Config.working_dir, "submission")
    Config.submission_path = os.path.join(Config.submission_dir, "submission.csv")

    # Use a tiny model for speed
    Config.model_name = "prajjwal1/bert-tiny"
    Config.debug = True  # Subsamples data (100 train, 50 test)
    Config.n_folds = 2  # Reduce folds
    Config.epochs = 1  # Reduce epochs
    Config.batch_size = 4
    Config.gradient_accumulation_steps = 1
    Config.setup()

    # Set seeds
    seed_everything(Config.seed)

    # -------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading and Preprocessing Data...")
    # This will load metadata, create folds, calc meta-features, and cache it
    train_df, test_df = load_and_preprocess(Config, load_cached_data=False)

    # Validation
    assert (
        len(train_df) == 100
    ), f"Expected 100 training samples in debug mode, got {len(train_df)}"
    assert (
        len(test_df) == 50
    ), f"Expected 50 test samples in debug mode, got {len(test_df)}"
    assert "fold" in train_df.columns, "Fold column missing in train_df"
    assert "word_count" in train_df.columns, "Meta-feature 'word_count' missing"
    print("Data loaded and verified.")

    # -------------------------------------------------------------------------
    # 3. Dataset & Model Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Dataset and Model Architecture...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create Dataset
    ds = EssayDataset(train_df.iloc[:4], tokenizer, Config, is_train=True)
    batch = ds[0]

    # Verify Dataset Output
    expected_keys = {
        "input_ids",
        "attention_mask",
        "meta_features",
        "essay_id",
        "labels",
    }
    assert expected_keys.issubset(
        batch.keys()
    ), f"Dataset missing keys. Found: {batch.keys()}"
    print("Dataset verification passed.")

    # Initialize Model
    device = Config.device
    model = EssayModel(pretrained=True).to(device)

    # Prepare batch
    input_ids = batch["input_ids"].unsqueeze(0).to(device)
    attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
    meta_features = batch["meta_features"].unsqueeze(0).to(device)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, meta_features)

    # Verify Model Output
    logits = outputs["logits"]
    embeddings = outputs["embeddings"]

    assert logits.shape == (1,), f"Expected logits shape (1,), got {logits.shape}"
    assert embeddings.shape == (
        1,
        128,
    ), f"Expected embeddings shape (1, 128) for bert-tiny, got {embeddings.shape}"
    print("Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. AWP Logic Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying AWP (Adversarial Weight Perturbation)...")
    # Create a dummy optimizer for AWP
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    awp = AWP(
        model, optimizer, adv_lr=1.0, adv_eps=2.0, scaler=scaler
    )  # High LR to ensure visible change

    # Save original weight of the classifier head
    orig_weight = model.fc.weight.data.clone()

    # Simulate a training step to generate gradients
    model.train()
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask, meta_features)
    loss = nn.MSELoss()(outputs["logits"], torch.tensor([3.0], device=device))
    loss.backward()

    # Attack
    awp.attack()
    attacked_weight = model.fc.weight.data.clone()

    # Verify weights changed
    diff = torch.norm(attacked_weight - orig_weight).item()
    assert diff > 1e-6, "AWP attack failed: Weights did not change."

    # Restore
    awp.restore()
    restored_weight = model.fc.weight.data.clone()

    # Verify weights restored
    restore_diff = torch.norm(restored_weight - orig_weight).item()
    assert restore_diff < 1e-6, "AWP restore failed: Weights not restored correctly."
    print("AWP logic verification passed.")

    # Clean up memory
    del model, optimizer, awp, scaler
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5. Backbone Training (Simulated)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Backbone Training (2 Folds)...")
    # train_backbone runs the loop for all folds defined in Config.n_folds
    # It saves models to checkpoints/ and OOF data to cache/
    train_backbone(train_df, Config)

    # Verify outputs
    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.model_dir, f"backbone_fold_{fold}.pth")
        oof_path = os.path.join(Config.cache_dir, f"oof_embeddings_fold_{fold}.npy")
        assert os.path.exists(model_path), f"Model for fold {fold} not found."
        assert os.path.exists(oof_path), f"OOF embeddings for fold {fold} not found."
    print("Backbone training complete.")

    # -------------------------------------------------------------------------
    # 6. Stacking Training
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running LightGBM Stacking...")
    # Trains LGBM on OOF embeddings + meta-features
    avg_score = train_stacking(train_df, Config)

    print(f"Stacking complete. Average QWK: {avg_score:.4f}")

    # Verify LGBM models
    for fold in range(Config.n_folds):
        lgbm_path = os.path.join(Config.model_dir, f"lgbm_fold_{fold}.txt")
        assert os.path.exists(lgbm_path), f"LGBM model for fold {fold} not found."

    # -------------------------------------------------------------------------
    # 7. Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Inference Pipeline...")
    # Uses the trained backbone and stacking models to predict on test_df
    submission = inference(test_df, Config)

    # Verify Submission
    assert os.path.exists(Config.submission_path), "Submission file not created."
    assert len(submission) == 50, f"Expected 50 predictions, got {len(submission)}"
    assert "essay_id" in submission.columns and "score" in submission.columns
    assert submission["score"].between(1, 6).all(), "Predictions out of range [1, 6]"

    print("\n=== Demo Completed Successfully ===")
    print(f"Submission saved to: {Config.submission_path}")
    print(submission.head())


if __name__ == "__main__":
    run_demo()
