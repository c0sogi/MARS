import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.dataset import get_dataloaders
from library.model import TripleBranchDistilRoBERTa
from library.engine import train_fn, eval_fn, get_optimizer_params


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # Set reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # --------------------------------------------------------------------------
    # 2. Data Loading & "Fast Baseline" Constraints
    # --------------------------------------------------------------------------
    # Check dataset size to ensure we meet the runtime limit
    try:
        train_meta = pd.read_csv(Config.TRAIN_PATH)
        n_samples = len(train_meta)
        print(f"Training data samples: {n_samples}")

        # If dataset is very large, reduce epochs to ensure fast execution
        if n_samples > 50000:
            print(
                "Large dataset detected. Reducing epochs to 3 to ensure runtime constraint."
            )
            Config.EPOCHS = 3
    except Exception as e:
        print(
            f"Warning: Could not check metadata size: {e}. Proceeding with default config."
        )

    # Load DataLoaders (utilizing cache)
    print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Model & Optimizer Initialization
    # --------------------------------------------------------------------------
    print(f"Initializing model on {device}...")
    model = TripleBranchDistilRoBERTa()
    model.to(device)

    # Configure Optimizer with differential learning rates
    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params)

    # Configure Scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
    )

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_score = -1.0
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Freeze/Unfreeze Logic
        if epoch == 0:
            print("Epoch 1: Freezing backbone layers for head warmup.")
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif epoch == 1:
            print("Epoch 2: Unfreezing backbone layers for fine-tuning.")
            for param in model.backbone.parameters():
                param.requires_grad = True

        # Train Step
        train_loss = train_fn(model, train_loader, optimizer, device, scheduler)

        # Validation Step
        val_loss, val_score = eval_fn(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Spearman: {val_score:.4f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with score: {best_score:.4f}")

    # --------------------------------------------------------------------------
    # 5. Final Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\nLoading best model for final validation and analysis...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)
    model.eval()

    # Generate predictions on validation set
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            title_ids = batch["title_input_ids"].to(device)
            title_mask = batch["title_attention_mask"].to(device)
            body_ids = batch["body_input_ids"].to(device)
            body_mask = batch["body_attention_mask"].to(device)
            answer_ids = batch["answer_input_ids"].to(device)
            answer_mask = batch["answer_attention_mask"].to(device)
            targets = batch["targets"].to(device)

            # Inference
            logits = model(
                title_ids, title_mask, body_ids, body_mask, answer_ids, answer_mask
            )
            preds = torch.sigmoid(logits)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Final Metric
    final_metric = compute_spearmanr(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    try:
        val_df = pd.read_csv(Config.VAL_PATH)

        # Ensure alignment
        if len(val_df) != len(val_preds):
            print(
                f"Warning: Validation DataFrame length ({len(val_df)}) matches predictions ({len(val_preds)})? Adjusting..."
            )
            min_len = min(len(val_df), len(val_preds))
            val_df = val_df.iloc[:min_len]
            val_preds = val_preds[:min_len]
            val_targets = val_targets[:min_len]

        # Calculate Mean Absolute Error per sample
        abs_err = np.abs(val_preds - val_targets)
        mean_abs_err = np.mean(abs_err, axis=1)

        # Calculate feature lengths
        val_df["title_len"] = val_df["question_title"].fillna("").str.len()
        val_df["body_len"] = val_df["question_body"].fillna("").str.len()
        val_df["answer_len"] = val_df["answer"].fillna("").str.len()

        # Correlate Error with Features
        print("Correlation between Error Magnitude and Input Features:")
        for feat in ["title_len", "body_len", "answer_len"]:
            corr, _ = spearmanr(val_df[feat], mean_abs_err)
            print(f"  Error vs {feat}: {corr:.4f}")

    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # --------------------------------------------------------------------------
    # 6. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.40802662717842303

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        all_preds = []
        all_qa_ids = []

        with torch.no_grad():
            for batch in test_loader:
                title_ids = batch["title_input_ids"].to(device)
                title_mask = batch["title_attention_mask"].to(device)
                body_ids = batch["body_input_ids"].to(device)
                body_mask = batch["body_attention_mask"].to(device)
                answer_ids = batch["answer_input_ids"].to(device)
                answer_mask = batch["answer_attention_mask"].to(device)
                qa_ids = batch["qa_ids"]

                logits = model(
                    title_ids, title_mask, body_ids, body_mask, answer_ids, answer_mask
                )
                preds = torch.sigmoid(logits)

                all_preds.append(preds.cpu().numpy())
                all_qa_ids.extend(qa_ids)

        if all_preds:
            final_preds = np.vstack(all_preds)

            # Create DataFrame
            sub_df = pd.DataFrame(final_preds, columns=Config.TARGET_COLS)
            sub_df.insert(0, "qa_id", all_qa_ids)

            # Save
            sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
        else:
            print("Error: No predictions generated for test set.")
    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
