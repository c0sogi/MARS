import os
import shutil
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, logging

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_loaders
from library.model import HybridDeberta
from library.trainer import train_one_epoch, valid_one_epoch, AWP
from library.inference import predict_fold, generate_submission

# Suppress HF warnings for cleaner output
logging.set_verbosity_error()


def run_demo():
    print("=== Starting Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Set seed for reproducibility
    seed_everything(Config.seed)

    # Override Config for speed and resource efficiency
    # We use a tiny model to avoid downloading/loading the large DeBERTa model
    Config.model_name = "prajjwal1/bert-tiny"
    Config.debug = True
    Config.debug_sample_size = 50  # Small sample for quick iteration
    Config.epochs = 1
    Config.n_folds = 1  # Only run one fold for demo
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.print_freq = 5
    Config.working_dir = "./working/demo_run"
    Config.output_dir = os.path.join(Config.working_dir, "models")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    print(f"    Model: {Config.model_name}")
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Working Dir: {Config.working_dir}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Pipeline (get_loaders)...")

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Force reload of cached data to ensure processing logic runs
    train_loader, val_loader, test_loader = get_loaders(
        tokenizer, load_cached_data=False
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Verify Train Batch Structure
    batch = next(iter(train_loader))
    required_keys = ["input_ids", "attention_mask", "features", "label", "score", "id"]
    for key in required_keys:
        assert key in batch, f"Missing key {key} in train batch"

    # Verify Shapes
    # input_ids: [Batch, Seq_Len]
    assert batch["input_ids"].shape[0] == Config.train_batch_size
    assert batch["input_ids"].shape[1] == Config.max_length
    # features: [Batch, Handcrafted_Dim]
    assert batch["features"].shape[1] == Config.handcrafted_features_dim
    # label: [Batch]
    assert batch["label"].shape[0] == Config.train_batch_size

    print("    Batch structure and shapes verified.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Testing Model Initialization and Forward Pass...")

    device = Config.device
    model = HybridDeberta()
    model.to(device)

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    features = batch["features"].to(device)

    # Forward pass
    logits = model(input_ids, attention_mask, features)

    # Check output shape: [Batch, Num_Classes]
    assert logits.shape == (Config.train_batch_size, Config.num_classes)
    print(
        f"    Logits Shape: {logits.shape} (Expected: {Config.train_batch_size}, {Config.num_classes})"
    )
    print("    Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Component
    # -------------------------------------------------------------------------
    print("\n[4] Testing Training Loop Components...")

    # Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = None  # Simplified for demo
    scaler = torch.cuda.amp.GradScaler()

    # Setup AWP
    awp = AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)

    # Run one epoch training
    print("    Running train_one_epoch...")
    train_loss = train_one_epoch(
        model,
        optimizer,
        scheduler,
        train_loader,
        device,
        epoch=1,
        awp=awp,
        scaler=scaler,
    )

    assert isinstance(train_loss, float)
    assert train_loss > 0
    print(f"    Training finished. Loss: {train_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Validation Loop Component
    # -------------------------------------------------------------------------
    print("\n[5] Testing Validation Loop Components...")

    print("    Running valid_one_epoch...")
    val_loss, val_pearson = valid_one_epoch(model, val_loader, device)

    assert isinstance(val_loss, float)
    assert isinstance(val_pearson, float)
    # Pearson can be negative, but usually between -1 and 1
    assert -1.0 <= val_pearson <= 1.0

    print(f"    Validation finished. Loss: {val_loss:.4f}, Pearson: {val_pearson:.4f}")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Testing Inference and Submission Generation...")

    # Save the model to simulate a trained fold
    fold_idx = 0
    model_path = os.path.join(Config.output_dir, f"model_fold_{fold_idx}.bin")
    torch.save(model.state_dict(), model_path)
    print(f"    Saved dummy model to {model_path}")

    # Test predict_fold
    print("    Testing predict_fold...")
    ids, scores = predict_fold(fold_idx, test_loader, device)

    assert len(ids) > 0
    assert len(ids) == len(scores)
    # Check against debug sample size (or actual test size if smaller)
    # In debug mode, test set is sampled to debug_sample_size
    expected_len = min(Config.debug_sample_size, 3648)  # 3648 is full test size
    # Note: If test set is smaller than debug_sample_size, it takes full.
    # Here we just verify we got predictions.
    print(f"    Generated {len(scores)} predictions.")

    # Test generate_submission
    # generate_submission internally calls predict_fold for all folds in range(Config.n_folds)
    # We set n_folds=1, so it will pick up the model we just saved.
    print("    Testing generate_submission...")
    generate_submission()

    # Verify file creation
    assert os.path.exists(Config.submission_path)
    df_sub = pd.read_csv(Config.submission_path)
    assert "id" in df_sub.columns
    assert "score" in df_sub.columns
    assert len(df_sub) > 0

    print(f"    Submission file verified at {Config.submission_path}")
    print(f"    Submission shape: {df_sub.shape}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
