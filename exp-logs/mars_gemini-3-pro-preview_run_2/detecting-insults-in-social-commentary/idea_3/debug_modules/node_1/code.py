import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_auc
from library.data import load_and_preprocess_data, get_dataloaders, get_test_dataloader
from library.model import InsultModel
from library.engine import get_optimizer_params, train_fn, eval_fn

# Suppress transformer warnings for cleaner output
logging.getLogger("transformers").setLevel(logging.ERROR)

if __name__ == "__main__":
    print("Starting demonstration script...")

    # 1. Configuration Setup
    # Override Config defaults for a fast demonstration run
    Config.working_dir = "./working/demo_run"
    Config.n_folds = 2
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.dropout_samples = 2  # Reduce dropout samples for speed

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set reproducibility
    set_seed(Config.seed)
    print(
        f"Configuration: Device={Config.device}, Epochs={Config.epochs}, Batch Size={Config.train_batch_size}"
    )

    # 2. Data Loading and Preprocessing
    # We force reprocessing (load_cached_data=False) to verify the tokenization logic
    print("\n[Data] Loading and preprocessing data...")
    train_df_full, test_df_full = load_and_preprocess_data(load_cached_data=False)

    # Validate output structure
    assert "input_ids" in train_df_full.columns, "train_df missing input_ids"
    assert "attention_mask" in train_df_full.columns, "train_df missing attention_mask"
    assert "fold" in train_df_full.columns, "train_df missing fold column"

    # Subsample data for speed demonstration
    # We take a small slice to ensure the training loop finishes in seconds
    subset_size = 40
    train_df_subset = train_df_full.head(subset_size).copy()
    test_df_subset = test_df_full.head(subset_size).copy()

    # Manually assign folds to the subset to ensure both train and val sets exist for fold 0
    # Assign first 80% to fold 1 (training set for fold 0 logic) and 20% to fold 0 (validation set)
    # Note: get_dataloaders(fold_idx=0) uses fold!=0 for training and fold==0 for validation
    train_df_subset["fold"] = 1
    val_indices = np.random.choice(
        train_df_subset.index, size=int(subset_size * 0.2), replace=False
    )
    train_df_subset.loc[val_indices, "fold"] = 0

    print(f"[Data] Subsampled training data to {len(train_df_subset)} rows.")
    print(
        f"[Data] Validation samples (fold 0): {len(train_df_subset[train_df_subset['fold'] == 0])}"
    )

    # 3. DataLoader Creation
    print("\n[Data] Creating DataLoaders...")
    train_loader, val_loader = get_dataloaders(train_df_subset, fold_idx=0)

    # Verify DataLoader yields correct batch structure
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "attention_mask" in sample_batch
    assert "labels" in sample_batch
    assert sample_batch["input_ids"].shape[0] == Config.train_batch_size
    print("[Data] DataLoaders created and verified successfully.")

    # 4. Model Initialization
    print("\n[Model] Initializing InsultModel (DeBERTa-v3-base)...")
    model = InsultModel()
    model.to(Config.device)

    # Verify forward pass with a dummy batch
    with torch.no_grad():
        dummy_out = model(
            sample_batch["input_ids"].to(Config.device),
            sample_batch["attention_mask"].to(Config.device),
        )
        assert dummy_out.shape == (
            Config.train_batch_size,
        ), f"Output shape mismatch: {dummy_out.shape}"
    print("[Model] Forward pass verification successful.")

    # 5. Optimizer and Scheduler Setup
    print("\n[Training] Setting up optimizer...")
    optimizer_parameters = get_optimizer_params(
        model, encoder_lr=2e-5, decoder_lr=1e-4, weight_decay=0.01
    )
    optimizer = AdamW(optimizer_parameters)

    # Simple linear scheduler
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # 6. Training Loop
    print(f"\n[Training] Starting training for {Config.epochs} epoch(s)...")
    for epoch in range(Config.epochs):
        train_loss = train_fn(train_loader, model, optimizer, Config.device, scheduler)
        print(f"  Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.4f}")

        # Verify loss is a valid number
        assert not np.isnan(train_loss), "Training loss is NaN"

    # 7. Evaluation
    print("\n[Evaluation] Evaluating on validation set...")
    val_loss, val_preds, val_targets = eval_fn(val_loader, model, Config.device)

    # Calculate AUC
    auc_score = calculate_auc(val_targets, val_preds)
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation AUC:  {auc_score:.4f}")

    # Assertions for evaluation
    assert len(val_preds) == len(val_targets)
    assert 0.0 <= auc_score <= 1.0, "AUC score out of range"

    # 8. Inference on Test Set
    print("\n[Inference] Generating predictions for test set...")
    test_loader = get_test_dataloader(test_df_subset)
    _, test_preds, _ = eval_fn(test_loader, model, Config.device)

    assert len(test_preds) == len(test_df_subset), "Prediction count mismatch"

    # Create submission dataframe
    submission = pd.DataFrame({"id": range(len(test_preds)), "prediction": test_preds})

    print("\n[Output] Sample predictions:")
    print(submission.head())

    print("\nDemonstration completed successfully.")
