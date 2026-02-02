import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from scipy.stats import spearmanr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import get_dataloaders
from library.model import ContextualDualEncoder
from library.train import train_epoch, validate_epoch, predict


def perform_failure_analysis(val_preds, val_targets, val_df):
    """
    Analyzes model failures by correlating error magnitude with input features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate Mean Absolute Error per sample (averaged across all 30 targets)
    # Shape: (N_samples,)
    sample_mae = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Extract features for correlation
    # We use text lengths as a proxy for complexity/information content
    features = {}

    if "question_title" in val_df.columns:
        features["q_title_len"] = val_df["question_title"].fillna("").str.len().values
    if "question_body" in val_df.columns:
        features["q_body_len"] = val_df["question_body"].fillna("").str.len().values
    if "answer" in val_df.columns:
        features["ans_len"] = val_df["answer"].fillna("").str.len().values

    print("Correlation between Error (MAE) and Input Features:")
    for name, feature_values in features.items():
        if len(feature_values) == len(sample_mae):
            corr, _ = spearmanr(sample_mae, feature_values)
            print(f"  Error vs {name}: {corr:.4f}")
        else:
            print(f"  Skipping {name} due to length mismatch.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    # We use the full dataset but will limit epochs for speed
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # 3. Model Initialization
    model = ContextualDualEncoder()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    best_score = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # 4. Training Loop

    # --- Phase 1: Warmup (Frozen Backbone) ---
    print("\n--- Phase 1: Warmup (Frozen Backbone) ---")
    model.freeze_backbone()

    optimizer_warmup = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR_HEAD,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Train 1 epoch warmup
    train_loss = train_epoch(
        model, train_loader, optimizer_warmup, None, criterion, device
    )
    val_loss, val_score = validate_epoch(model, val_loader, criterion, device)
    print(
        f"Warmup Epoch - Train Loss: {train_loss:.4f} - Val Spearman: {val_score:.4f}"
    )

    best_score = val_score
    torch.save(model.state_dict(), best_model_path)

    # --- Phase 2: Fine-tuning (Unfrozen Backbone) ---
    print("\n--- Phase 2: Fine-tuning (Unfrozen Backbone) ---")
    model.unfreeze_backbone()

    # Differential learning rates
    optimizer_grouped_parameters = [
        {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
        {"params": model.head.parameters(), "lr": Config.LR_HEAD},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=Config.WEIGHT_DECAY)

    # Increase to 5 epochs to allow full convergence
    finetune_epochs = 5
    total_steps = len(train_loader) * finetune_epochs
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    for epoch in range(finetune_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_score = validate_epoch(model, val_loader, criterion, device)

        print(
            f"Finetune Epoch {epoch+1}/{finetune_epochs} - Train Loss: {train_loss:.4f} - Val Spearman: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Validation Assessment
    print("\n--- Final Validation Assessment ---")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # We need predictions and targets for failure analysis
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            probs = torch.sigmoid(logits)

            val_preds_list.append(probs.cpu().numpy())
            val_targets_list.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    final_metric = compute_spearman_metric(val_preds, val_targets)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    val_df = pd.read_csv(Config.VAL_PATH)
    perform_failure_analysis(val_preds, val_targets, val_df)

    # 7. Submission Logic
    # Lower threshold to account for stochastic variance (Cite solution_lesson_node_00010)
    THRESHOLD = 0.395

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        test_preds = predict(model, test_loader, device)

        # Load test metadata for IDs
        test_df = pd.read_csv(Config.TEST_PATH)

        # Create submission DataFrame
        submission = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
        submission.insert(0, "qa_id", test_df["qa_id"])

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
