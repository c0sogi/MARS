import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from scipy.stats import spearmanr

# Import library components
from library.config import config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DebertaDualEncoder
from library.train import train_fn, eval_fn, get_optimizer_params, inference_fn


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    seed_everything(config.seed)
    device = torch.device(config.device)

    # Override epochs for a fast baseline execution
    config.epochs = 3

    print(f"Device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Initializing Tokenizer and Loading Data...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_loader, val_loader, test_loader, meta_dims = get_dataloaders(
        config, tokenizer, load_cached_data=True
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing Model...")
    model = DebertaDualEncoder(meta_dims)
    model.to(device)

    # ==========================================
    # 4. Optimizer & Scheduler
    # ==========================================
    optimizer_parameters = get_optimizer_params(model)
    optimizer = AdamW(
        optimizer_parameters,
        lr=config.lr_head,
        eps=config.eps,
        betas=config.betas,
    )

    num_train_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_train_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    criterion = nn.BCEWithLogitsLoss()

    # ==========================================
    # 5. Training Loop
    # ==========================================
    best_score = -1.0
    print(f"Starting Training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_score, _ = eval_fn(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.4f} | Val Spearman: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.model_save_path)

    # ==========================================
    # 6. Final Validation & Metric
    # ==========================================
    print("\n--- Final Evaluation ---")
    # Load best model for evaluation
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    model.eval()

    # Get predictions on validation set
    val_loss, final_metric, val_preds = eval_fn(model, val_loader, criterion, device)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 7. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")

    # Load validation dataframe to get features (using cached processed data)
    val_parquet_path = os.path.join(config.working_dir, "val_processed.parquet")
    if os.path.exists(val_parquet_path):
        val_df = pd.read_parquet(val_parquet_path)
    else:
        # Fallback if parquet not found (should not happen if get_dataloaders worked)
        val_df = pd.read_csv(config.val_path)
        # Re-apply basic processing for analysis
        val_df["question_text"] = (
            val_df["question_title"].fillna("")
            + " "
            + val_df["question_body"].fillna("")
        )
        val_df["answer"] = val_df["answer"].fillna("")

    # Align predictions with dataframe
    # Dataloader is sequential, but ensure lengths match
    if len(val_df) != len(val_preds):
        min_len = min(len(val_df), len(val_preds))
        val_df = val_df.iloc[:min_len]
        val_preds = val_preds[:min_len]

    # Ground truth
    y_true = val_df[config.target_cols].values

    # Compute Mean Absolute Error per sample across all targets
    errors = np.abs(y_true - val_preds).mean(axis=1)
    val_df["error_magnitude"] = errors

    # Compute features for correlation
    val_df["q_len"] = val_df["question_text"].str.len()
    val_df["a_len"] = val_df["answer"].str.len()

    # Features to analyze
    analysis_features = ["q_len", "a_len"]

    print("Correlation between Error Magnitude and Input Features:")
    for feat in analysis_features:
        if feat in val_df.columns:
            corr, _ = spearmanr(val_df["error_magnitude"], val_df[feat])
            print(f"  {feat}: {corr:.4f}")

    # ==========================================
    # 8. Conditional Submission
    # ==========================================
    THRESHOLD = 0.40802662717842303

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference
        test_preds = inference_fn(model, test_loader, device)

        # Load test metadata for IDs
        test_df_meta = pd.read_csv(config.test_path)

        # Align lengths in case of debug truncation
        if len(test_preds) != len(test_df_meta):
            test_df_meta = test_df_meta.iloc[: len(test_preds)]

        # Create submission dataframe
        submission = pd.DataFrame(test_preds, columns=config.target_cols)
        submission.insert(0, "qa_id", test_df_meta["qa_id"].values)

        # Save
        submission.to_csv(config.submission_path, index=False)
        print(f"Submission saved to {config.submission_path}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
