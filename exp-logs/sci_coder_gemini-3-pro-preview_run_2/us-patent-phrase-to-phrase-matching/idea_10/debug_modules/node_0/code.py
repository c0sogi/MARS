import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, preprocess_data, PearsonDataset, collate_fn
from library.model import CustomDeberta
from library.awp import AWP
from library.engine import train_fn, eval_fn
from library.stacking import train_lgbm_stacker


def run_demo():
    print("=== Phrase Similarity Model Demo ===")

    # 1. Configuration Overrides for Speed/Demo
    # We enable debug mode to use only 100 samples per split
    Config.debug = True
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.gradient_accumulation_steps = 1

    # Set up a specific working directory for this demo
    Config.working_dir = "./working/demo_run"
    Config.model_output_dir = os.path.join(Config.working_dir, "models")
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Create directories
    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs(Config.model_output_dir, exist_ok=True)
    os.makedirs(Config.cache_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.seed)
    device = Config.device
    print(f"Device: {device}")
    print(f"Debug Mode: {Config.debug}")

    # 2. Data Preparation
    print("\n[Step 1] Preparing Data...")

    # Initialize Tokenizer
    # We use the model name defined in Config (microsoft/deberta-v3-large)
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Get DataLoaders (Train, Val, Test)
    # load_cached_data=False forces reprocessing to ensure we use the debug subset
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify Batch Structure
    batch = next(iter(train_loader))
    assert "input_ids" in batch, "Batch missing input_ids"
    assert "structural_features" in batch, "Batch missing structural_features"
    assert "labels" in batch, "Batch missing labels"
    print("Batch structure verified.")

    # 3. Model Initialization
    print("\n[Step 2] Initializing Model...")
    model = CustomDeberta()
    model.to(device)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Initialize Scheduler
    num_training_steps = len(train_loader) * Config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * Config.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    # 4. AWP Initialization
    print("\n[Step 3] Initializing Adversarial Weight Perturbation (AWP)...")
    awp = AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)

    # 5. Training Loop
    print("\n[Step 4] Starting Training (1 Epoch)...")
    # We run epoch 0. AWP starts at Config.awp_start_epoch (default 1),
    # so we temporarily lower it to 0 to demonstrate AWP execution.
    Config.awp_start_epoch = 0

    train_loss = train_fn(
        train_loader,
        model,
        optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        awp=awp,
    )
    print(f"Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss resulted in NaN"

    # 6. Evaluation
    print("\n[Step 5] Evaluating on Validation Set...")
    val_loss, val_preds, val_score = eval_fn(val_loader, model, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Pearson Score: {val_score:.4f}")

    # Verify predictions count (Debug mode = 100 samples)
    assert (
        len(val_preds) == 100
    ), f"Expected 100 validation predictions, got {len(val_preds)}"

    # 7. Stacking Preparation
    print("\n[Step 6] Preparing Features for Stacking...")

    # The stacker requires predictions for Train, Val, and Test.
    # The Train predictions must be aligned with the metadata order.
    # The standard train_loader is shuffled, so we create a sequential one here.

    # Load the processed train dataframe (cached by get_dataloaders)
    train_df = preprocess_data("train", load_cached_data=True)

    # Create a non-shuffled dataset/loader
    train_ds_ordered = PearsonDataset(train_df, tokenizer, is_train=False)
    train_loader_ordered = DataLoader(
        train_ds_ordered,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
    )

    # Generate predictions
    print("Generating aligned training predictions...")
    _, train_preds_aligned, _ = eval_fn(train_loader_ordered, model, device)

    print("Generating test predictions...")
    _, test_preds, _ = eval_fn(test_loader, model, device)

    # 8. Stacking Execution
    print("\n[Step 7] Running LightGBM Stacker...")
    # This trains the meta-learner and saves the submission file
    # We disable cache loading for features to ensure they match our current debug run
    train_lgbm_stacker(
        train_preds=train_preds_aligned,
        val_preds=val_preds,
        test_preds=test_preds,
        load_cached_data=False,
    )

    # 9. Final Verification
    print("\n[Step 8] Verifying Submission...")
    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    sub_df = pd.read_csv(Config.submission_path)
    print(f"Submission File: {Config.submission_path}")
    print(sub_df.head())
    print(f"Shape: {sub_df.shape}")

    # In debug mode, test set is 100 rows
    assert sub_df.shape == (100, 2), f"Expected shape (100, 2), got {sub_df.shape}"
    assert list(sub_df.columns) == ["id", "score"], "Incorrect columns in submission"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
