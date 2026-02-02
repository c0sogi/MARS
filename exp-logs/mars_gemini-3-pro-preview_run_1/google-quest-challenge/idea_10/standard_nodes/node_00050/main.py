import os
import torch
import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from scipy.stats import spearmanr

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DistilRobertaDualEncoder
from library.engine import get_optimizer_params, train_fn, eval_fn, generate_submission


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # Config.DEBUG is False by default, so this loads the full dataset (~4.4k train, ~1k val)
    # This is small enough to train quickly (approx 5-10 mins on A100)
    train_loader, val_loader, test_loader, target_cols = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    model = DistilRobertaDualEncoder()
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params)

    num_train_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # 5. Training Loop
    best_score = -1.0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)

        # Validate
        val_loss, val_score = eval_fn(val_loader, model, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Spearman: {val_score:.4f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with score: {best_score:.4f}")

    # 6. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    # We re-verify the metric on the loaded model (sanity check)
    _, final_score = eval_fn(val_loader, model, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []

    # Generate validation predictions
    with torch.no_grad():
        for batch in val_loader:
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Mean Absolute Error (MAE) per sample averaged across all 30 targets
    mae_per_sample = np.mean(np.abs(all_preds - all_targets), axis=1)

    # Retrieve metadata for correlation analysis
    # val_loader.dataset.df contains the validation dataframe in order
    val_df = val_loader.dataset.df

    # Calculate lengths
    q_lens = val_df["question_text"].fillna("").str.len().values
    a_lens = val_df["answer"].fillna("").str.len().values

    # Compute correlations
    corr_q, _ = spearmanr(mae_per_sample, q_lens)
    corr_a, _ = spearmanr(mae_per_sample, a_lens)

    print(f"Correlation between Error (MAE) and Question Length: {corr_q:.4f}")
    print(f"Correlation between Error (MAE) and Answer Length: {corr_a:.4f}")

    # 8. Submission
    THRESHOLD = 0.40802662717842303

    if final_score > THRESHOLD:
        print(
            f"\nValidation score ({final_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(test_loader, model, device)
    else:
        print(
            f"\nValidation score ({final_score}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
