import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from scipy.stats import spearmanr
import os
import sys

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.data import get_dataloaders, get_target_columns
from library.model import QuestModel
from library.engine import train_fn, eval_fn, predict_fn, get_optimizer_params


def main():
    # 1. Setup Environment
    seed_everything(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Initialize Model
    print(f"Initializing model: {Config.model_name}")
    model = QuestModel()
    model.to(device)

    # 4. Optimizer and Scheduler
    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params, eps=Config.eps, betas=Config.betas)

    num_train_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_score = -1.0
    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, epoch, device
        )

        # Validate (Score calculation is inside eval_fn)
        val_loss, val_score = eval_fn(val_loader, model, criterion, device)

        print(
            f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Score={val_score:.4f}"
        )

        # Save Best Model
        if val_score > best_score:
            print(
                f"Score Improved ({best_score:.4f} -> {val_score:.4f}). Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)

    # 6. Final Evaluation & Failure Analysis
    print("\nLoading best model for final evaluation and analysis...")
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    model.eval()

    # Generate predictions on validation set for detailed analysis
    val_preds = []
    val_targets = []
    val_qa_ids = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            labels = batch["labels"].to(device)
            qa_ids = batch["qa_ids"]

            logits = model(
                input_ids_q=input_ids_q,
                attention_mask_q=attention_mask_q,
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
            )
            preds = torch.sigmoid(logits)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(labels.cpu().numpy())
            val_qa_ids.extend(qa_ids.numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    # Compute Final Metric
    final_metric = compute_spearmanr(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")

    # Load validation metadata to get text features
    val_df = pd.read_csv(Config.val_path)

    # Calculate error per sample (Mean Absolute Error across all 30 targets)
    # Shape: (N_samples,)
    mae_per_sample = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Create analysis dataframe
    analysis_df = pd.DataFrame({"qa_id": val_qa_ids, "error": mae_per_sample})

    # Merge with text features from metadata
    # Calculate lengths
    val_df["q_len"] = (
        val_df[Config.question_title_col].fillna("")
        + " "
        + val_df[Config.question_body_col].fillna("")
    ).str.len()
    val_df["a_len"] = val_df[Config.answer_col].fillna("").str.len()

    # Merge
    analysis_df = analysis_df.merge(
        val_df[["qa_id", "q_len", "a_len"]], on="qa_id", how="left"
    )

    # Compute correlations
    corr_q, _ = spearmanr(analysis_df["error"], analysis_df["q_len"])
    corr_a, _ = spearmanr(analysis_df["error"], analysis_df["a_len"])

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Question Length vs Error: {corr_q:.4f}")
    print(f"  Answer Length vs Error:   {corr_a:.4f}")

    # 7. Conditional Submission
    THRESHOLD = 0.40802662717842303

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test
        qa_ids, predictions = predict_fn(test_loader, model, device)

        # Format Submission
        target_cols = get_target_columns()
        submission_df = pd.DataFrame(predictions, columns=target_cols)
        submission_df.insert(0, Config.qa_id_col, qa_ids)

        # Ensure qa_id is int
        submission_df[Config.qa_id_col] = submission_df[Config.qa_id_col].astype(int)

        # Save
        submission_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
