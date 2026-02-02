import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from transformers import get_linear_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.dataset import get_dataloaders
from library.model import SymmetricDualEncoder
from library.engine import train_fn, eval_fn, inference_fn


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True as requested
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # 3. Model Initialization
    model = SymmetricDualEncoder()
    model.to(device)

    # 4. Optimizer & Scheduler Setup
    # Group parameters for differential learning rates
    backbone_params = list(model.q_encoder.parameters()) + list(
        model.a_encoder.parameters()
    )
    head_params = (
        list(model.alignment_bridge.parameters())
        + list(model.layer_norm.parameters())
        + list(model.head_proj.parameters())
        + list(model.final_proj.parameters())
    )

    optimizer_parameters = [
        {"params": backbone_params, "lr": Config.LR_BACKBONE},
        {"params": head_params, "lr": Config.LR_HEAD},
    ]

    optimizer = torch.optim.AdamW(optimizer_parameters)

    # Phantom Scheduling
    num_train_steps = int(
        len(train_loader) / Config.ACCUMULATION_STEPS * Config.PHANTOM_EPOCHS
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    # 5. Training Loop
    best_score = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.ACTUAL_EPOCHS} epochs...")

    for epoch in range(Config.ACTUAL_EPOCHS):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler, epoch)

        # Validate
        val_loss, val_score, _ = eval_fn(val_loader, model, device)

        print(
            f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Score={val_score:.4f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Validation & Metric
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Get predictions on validation set for metrics and failure analysis
    val_loss, final_val_score, val_preds = eval_fn(val_loader, model, device)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_val_score}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    try:
        # Load validation metadata to get features
        val_df = pd.read_csv(Config.VAL_PATH)

        # Ensure alignment
        if len(val_df) != len(val_preds):
            print(
                "Warning: Validation dataframe length mismatch with predictions. Skipping detailed analysis."
            )
        else:
            # Get ground truth
            targets = val_df[Config.TARGET_COLS].values

            # Calculate Mean Absolute Error per sample (averaged across 30 targets)
            # Handle potential NaNs in targets if any (though dataset info says 0 nans)
            error_per_sample = np.nanmean(np.abs(targets - val_preds), axis=1)

            # Extract features
            # Feature 1: Question Body Length
            q_lens = val_df["question_body"].fillna("").str.len().values
            # Feature 2: Answer Length
            a_lens = val_df["answer"].fillna("").str.len().values

            # Compute correlations
            corr_q = spearmanr(error_per_sample, q_lens).statistic
            corr_a = spearmanr(error_per_sample, a_lens).statistic

            print(f"Correlation between Error and Question Body Length: {corr_q:.4f}")
            print(f"Correlation between Error and Answer Length: {corr_a:.4f}")

            # Identify worst performing target
            col_errors = np.nanmean(np.abs(targets - val_preds), axis=0)
            worst_col_idx = np.argmax(col_errors)
            print(
                f"Target with highest MAE: {Config.TARGET_COLS[worst_col_idx]} (MAE: {col_errors[worst_col_idx]:.4f})"
            )

    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # 8. Conditional Submission
    THRESHOLD = 0.4113257391193607

    if final_val_score > THRESHOLD:
        print(
            f"\nValidation score ({final_val_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds, test_ids = inference_fn(test_loader, model, device)

        submission = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
        submission.insert(0, "qa_id", test_ids)

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score ({final_val_score}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
