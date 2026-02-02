import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import warnings

# Add the current directory to sys.path to ensure library modules are found
sys.path.append(".")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import create_dataloaders
from library.modeling import ToxicityModel
from library.engine import train_fn, valid_fn, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("=== Starting Toxicity Classification Demo ===\n")

    # 1. Configuration
    # We use debug=True to limit data to 1000 samples for speed.
    # We set epochs=1 to verify the loop runs without waiting for full training.
    print("Initializing Configuration...")
    cfg = Config(debug=True, epochs=1, batch_size=8)

    # Override working directory for this demo
    cfg.working_dir = "./working/demo_execution"
    if os.path.exists(cfg.working_dir):
        shutil.rmtree(cfg.working_dir)
    os.makedirs(cfg.working_dir, exist_ok=True)

    # Update paths in config based on new working dir
    cfg.model_save_path = os.path.join(cfg.working_dir, "model.pth")
    cfg.submission_path = os.path.join(cfg.working_dir, "submission.csv")

    # Enable AWP from epoch 0 to verify it runs
    cfg.use_awp = True
    cfg.awp_start_epoch = 0

    # Set seed for reproducibility
    seed_everything(cfg.seed)
    print(f"Config: {cfg}\n")

    # 2. Data Loading
    print("Creating DataLoaders (processing from scratch)...")
    # load_cached_data=False forces the processing logic to run
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, load_cached_data=False
    )

    # Verification: Check DataLoader output
    print("Verifying DataLoader shapes...")
    sample_batch = next(iter(train_loader))
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    labels = sample_batch["labels"]

    assert input_ids.shape == (
        cfg.train_batch_size,
        cfg.max_len,
    ), f"Expected input_ids shape {(cfg.train_batch_size, cfg.max_len)}, got {input_ids.shape}"
    assert labels.shape == (
        cfg.train_batch_size,
        cfg.num_classes,
    ), f"Expected labels shape {(cfg.train_batch_size, cfg.num_classes)}, got {labels.shape}"
    print("DataLoader verification passed.\n")

    # 3. Model Initialization
    print("Initializing Model...")
    device = cfg.device
    model = ToxicityModel(cfg, pretrained=True)
    model.to(device)

    # Verification: Dummy Forward Pass
    print("Verifying Model Forward Pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids.to(device), attention_mask.to(device), labels.to(device)
        )

    logits = outputs["logits"]
    loss = outputs["loss"]

    assert logits.shape == (
        cfg.train_batch_size,
        cfg.num_classes,
    ), f"Expected logits shape {(cfg.train_batch_size, cfg.num_classes)}, got {logits.shape}"
    assert loss is not None and not torch.isnan(loss), "Model returned None or NaN loss"
    print("Model forward pass verification passed.\n")

    # 4. Training Loop Setup
    print("Setting up Optimizer and Scheduler...")
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    num_train_steps = len(train_loader) * cfg.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=num_train_steps, pct_start=cfg.pct_start
    )

    # 5. Run Training Epoch
    print("Running Training Epoch 1 (with AWP)...")
    train_loss = train_fn(
        train_loader, model, optimizer, scheduler, device, epoch=0, cfg=cfg
    )
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # 6. Run Validation
    print("Running Validation...")
    val_loss, val_score, val_preds = valid_fn(val_loader, model, device, cfg)
    print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_score:.4f}")

    assert 0 <= val_score <= 1, f"AUC Score {val_score} is out of bounds [0, 1]"
    assert (
        val_preds.shape[1] == cfg.num_classes
    ), "Validation predictions have incorrect class count"

    # Save model manually to simulate the training loop saving the best model
    torch.save(model.state_dict(), cfg.model_save_path)
    print(f"Model saved to {cfg.model_save_path}\n")

    # 7. Inference & Submission
    print("Generating Submission...")
    generate_submission(cfg, model, test_loader, device)

    # Verification: Check Submission File
    print("Verifying Submission File...")
    if not os.path.exists(cfg.submission_path):
        raise FileNotFoundError(f"Submission file not found at {cfg.submission_path}")

    submission_df = pd.read_csv(cfg.submission_path)
    sample_sub = pd.read_csv(cfg.sample_submission_path)

    if cfg.debug:
        # In debug mode, we only predict a subset, so align the reference
        sample_sub = sample_sub.iloc[: len(submission_df)]

    # Check dimensions
    assert len(submission_df) == len(
        sample_sub
    ), f"Submission rows {len(submission_df)} mismatch sample {len(sample_sub)}"

    # Check columns
    expected_cols = ["id"] + cfg.target_cols
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    # Check value range
    pred_values = submission_df[cfg.target_cols].values
    assert (pred_values >= 0).all() and (
        pred_values <= 1
    ).all(), "Prediction probabilities are out of range [0, 1]"

    print(f"Submission file verified successfully. Shape: {submission_df.shape}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
