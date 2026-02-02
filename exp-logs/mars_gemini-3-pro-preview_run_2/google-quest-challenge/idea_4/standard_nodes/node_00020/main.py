import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config, set_seed
from library.model import UnifiedDebertaSiamese
from library.dataset import get_dataloader
from library.engine import train_one_epoch, validate, predict_and_submit
from library.utils import compute_spearman_metric


def get_predictions(model, dataloader, device):
    """
    Runs inference on a dataloader and returns predictions and targets.
    """
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)

            if "labels" in batch:
                labels = batch["labels"].to(device)
                targets_list.append(labels.cpu().numpy())

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu().numpy())

    predictions = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0) if targets_list else None
    return predictions, targets


def perform_failure_analysis(predictions, targets, val_df):
    """
    Analyzes model errors and correlates them with input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Mean Absolute Error per sample
    # predictions: (N, 30), targets: (N, 30)
    errors = np.abs(predictions - targets)
    mean_errors = np.mean(errors, axis=1)  # (N,)

    # Extract features for correlation from the validation dataframe
    # We construct lengths based on the raw text
    val_df["q_len"] = (
        val_df["question_title"].fillna("") + " " + val_df["question_body"].fillna("")
    ).apply(len)
    val_df["a_len"] = val_df["answer"].fillna("").apply(len)

    # Compute correlations
    corr_q, _ = spearmanr(mean_errors, val_df["q_len"])
    corr_a, _ = spearmanr(mean_errors, val_df["a_len"])

    print(f"Correlation between Error Magnitude and Question Length: {corr_q:.4f}")
    print(f"Correlation between Error Magnitude and Answer Length: {corr_a:.4f}")

    # Identify worst performing target
    col_errors = np.mean(errors, axis=0)
    worst_col_idx = np.argmax(col_errors)
    worst_col_name = Config.TARGET_COLS[worst_col_idx]
    print(
        f"Target with highest mean error: {worst_col_name} (MAE: {col_errors[worst_col_idx]:.4f})"
    )


def main():
    # 1. Configuration & Setup
    # Override Config for fast baseline execution while maintaining performance
    Config.EPOCHS = 6

    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 2. Data Preparation
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load DataLoaders (using cached data if available)
    print("Preparing DataLoaders...")
    train_loader = get_dataloader("train", tokenizer, load_cached_data=True)
    val_loader = get_dataloader("val", tokenizer, load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = UnifiedDebertaSiamese()
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer_grouped_parameters = Config.get_optimizer_params(model)
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, eps=Config.EPS, betas=Config.BETAS
    )

    criterion = nn.BCEWithLogitsLoss()

    num_train_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # 5. Training Loop
    best_score = -1.0
    save_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )

        val_loss, val_score = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Spearman: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            print(f"New best score! Model saved.")

    print(f"Training finished. Best Score: {best_score:.4f}")

    # 6. Evaluation & Failure Analysis
    print("Loading best model for final evaluation and analysis...")
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Get predictions on validation set
    val_preds, val_targets = get_predictions(model, val_loader, device)

    # Compute Final Metric
    final_metric = compute_spearman_metric(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # Load Validation DataFrame for analysis
    val_df = pd.read_csv(Config.VAL_PATH)

    # Perform Failure Analysis
    perform_failure_analysis(val_preds, val_targets, val_df)

    # 7. Conditional Submission
    THRESHOLD = 0.39609456952678757

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, tokenizer)
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
