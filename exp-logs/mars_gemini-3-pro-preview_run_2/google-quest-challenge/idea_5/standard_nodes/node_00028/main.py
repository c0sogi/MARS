import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.stats import spearmanr
import os

from library.config import Config
from library.utils import seed_everything, compute_spearman_metric, save_checkpoint
from library.dataset import get_dataloaders
from library.model import HybridDeberta
from library.trainer import train_fn, eval_fn, predict_fn


def main():
    # ==========================================
    # 1. Configuration Overrides for Fast Baseline
    # ==========================================
    # Reduce epochs to ensure execution within time limits while allowing convergence
    Config.epochs = 4

    # ==========================================
    # 2. Setup
    # ==========================================
    seed_everything(Config.seed)
    device = Config.device
    print(f"Running on device: {device}")

    # ==========================================
    # 3. Data Loading
    # ==========================================
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    print("Initializing HybridDeberta Model...")
    model = HybridDeberta()
    model.to(device)

    # ==========================================
    # 5. Optimizer & Scheduler
    # ==========================================
    # Differential Learning Rates
    optimizer_grouped_parameters = [
        {"params": model.backbone.parameters(), "lr": Config.lr_backbone},
        {"params": model.head_intrinsic.parameters(), "lr": Config.lr_head},
        {"params": model.head_relational.parameters(), "lr": Config.lr_head},
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters, weight_decay=Config.weight_decay
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.epochs, T_mult=1, eta_min=Config.min_lr
    )

    criterion = nn.BCEWithLogitsLoss()

    # ==========================================
    # 6. Training Loop
    # ==========================================
    best_score = -1.0
    best_model_path = Config.model_save_path

    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss, val_score = eval_fn(val_loader, model, criterion, device)

        # Step Scheduler (Epoch-based)
        scheduler.step()

        print(
            f"Epoch {epoch+1} | Val Loss: {val_loss:.6f} | Val Spearman: {val_score:.6f}"
        )

        # Save Best Model
        if val_score > best_score:
            print(
                f"Score Improved ({best_score:.6f} -> {val_score:.6f}). Saving model..."
            )
            best_score = val_score
            save_checkpoint(model, best_model_path)

    # ==========================================
    # 7. Final Validation & Metric Calculation
    # ==========================================
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Running inference on validation set...")
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            view1_input_ids = batch["view1_input_ids"].to(device)
            view1_attention_mask = batch["view1_attention_mask"].to(device)
            view2_input_ids = batch["view2_input_ids"].to(device)
            view2_attention_mask = batch["view2_attention_mask"].to(device)
            view2_q_mask = batch["view2_q_mask"].to(device)
            view2_a_mask = batch["view2_a_mask"].to(device)
            labels = batch["labels"].to(device)

            view2_token_type_ids = None
            if "view2_token_type_ids" in batch:
                view2_token_type_ids = batch["view2_token_type_ids"].to(device)

            logits = model(
                view1_input_ids=view1_input_ids,
                view1_attention_mask=view1_attention_mask,
                view2_input_ids=view2_input_ids,
                view2_attention_mask=view2_attention_mask,
                view2_q_mask=view2_q_mask,
                view2_a_mask=view2_a_mask,
                view2_token_type_ids=view2_token_type_ids,
            )

            probs = torch.sigmoid(logits)
            val_preds_list.append(probs.cpu().numpy())
            val_targets_list.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    final_metric = compute_spearman_metric(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 8. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")
    # Load validation metadata to get text features
    val_df = pd.read_csv(Config.val_path)

    # Ensure alignment (dataset loader drops last batch in training, but validation is full)
    # The validation loader does not drop last, so sizes should match.
    if len(val_df) != len(val_preds):
        print(
            f"Warning: Validation DF size ({len(val_df)}) != Predictions size ({len(val_preds)})"
        )
        # Truncate to match if necessary (though shouldn't happen with correct config)
        min_len = min(len(val_df), len(val_preds))
        val_df = val_df.iloc[:min_len]
        val_preds = val_preds[:min_len]
        val_targets = val_targets[:min_len]

    # Calculate Mean Absolute Error per sample
    abs_errors = np.abs(val_preds - val_targets)
    mean_abs_error = np.mean(abs_errors, axis=1)

    # Compute Length Features
    # Fill NA just in case
    q_text = (
        val_df["question_title"].fillna("") + " " + val_df["question_body"].fillna("")
    ).astype(str)
    a_text = val_df["answer"].fillna("").astype(str)

    val_df["q_len"] = q_text.apply(len)
    val_df["a_len"] = a_text.apply(len)

    # Compute Correlations
    corr_q, _ = spearmanr(mean_abs_error, val_df["q_len"])
    corr_a, _ = spearmanr(mean_abs_error, val_df["a_len"])

    print(f"Correlation between Error and Question Length: {corr_q:.6f}")
    print(f"Correlation between Error and Answer Length: {corr_a:.6f}")

    # ==========================================
    # 9. Submission Generation
    # ==========================================
    threshold = 0.41003785424660755

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Generate predictions on test set
        test_preds = predict_fn(test_loader, model, device)

        # Load test metadata for IDs
        test_df = pd.read_csv(Config.test_path)

        # Create Submission DataFrame
        sub_df = pd.DataFrame(test_preds, columns=Config.target_cols)
        sub_df.insert(0, "qa_id", test_df["qa_id"])

        # Save
        sub_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
