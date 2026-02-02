import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import os
import copy
import time
from scipy.stats import spearmanr
from transformers import get_linear_schedule_with_warmup

# Import from library
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.data import get_data_loaders
from library.model import GranularSiameseDeBERTa
from library.train import train_fn, eval_fn


def run():
    # ==========================================
    # 1. Setup & Initialization
    # ==========================================
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    # Utilize pre-processed/cached data for speed
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # ==========================================
    # 3. Model & Optimizer Setup
    # ==========================================
    print("Initializing model...")
    model = GranularSiameseDeBERTa()
    model.to(device)

    # Differential Learning Rates
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": Config.LR_BACKBONE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": Config.LR_HEAD,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters)

    # Scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_score = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train one epoch
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)

        # Validate
        val_loss, val_preds, val_targets = eval_fn(model, val_loader, device)
        val_score = compute_spearmanr(val_preds, val_targets)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Time: {elapsed:.0f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Spearman: {val_score:.6f}"
        )

        # Checkpointing
        if val_score > best_score:
            print(
                f"Validation score improved ({best_score:.6f} -> {val_score:.6f}). Saving weights..."
            )
            best_score = val_score
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save to disk as well for persistence
            torch.save(
                best_model_wts, os.path.join(Config.WORKING_DIR, "best_model.pth")
            )
            patience = 0
        else:
            patience += 1
            print(
                f"No improvement. Patience: {patience}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # ==========================================
    # 5. Final Evaluation
    # ==========================================
    print("Loading best model for final evaluation...")
    model.load_state_dict(best_model_wts)

    print("Running final validation...")
    _, val_preds, val_targets = eval_fn(model, val_loader, device)
    final_metric = compute_spearmanr(val_preds, val_targets)

    # REQUIRED: Print full precision metric
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")
    # Load raw validation data to get text lengths for correlation analysis
    val_df = pd.read_csv(Config.VAL_PATH)

    # Verify alignment
    if len(val_df) != len(val_preds):
        print(
            f"Warning: Validation DF length ({len(val_df)}) != Predictions length ({len(val_preds)})"
        )

    # Calculate row-wise Mean Absolute Error (MAE)
    # val_targets and val_preds are (N, 30)
    row_errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Extract features for analysis
    val_df["q_len_char"] = val_df["question_body"].fillna("").astype(str).apply(len)
    val_df["a_len_char"] = val_df["answer"].fillna("").astype(str).apply(len)
    val_df["q_title_len_char"] = (
        val_df["question_title"].fillna("").astype(str).apply(len)
    )

    features_to_check = ["q_len_char", "a_len_char", "q_title_len_char"]
    print("Correlation between Mean Absolute Error and Input Features:")

    for feat in features_to_check:
        if feat in val_df.columns:
            # Handle scipy version differences for spearmanr return type
            res = spearmanr(row_errors, val_df[feat])
            corr = res.statistic if hasattr(res, "statistic") else res[0]
            print(f"  Error vs {feat}: {corr:.4f}")

    # ==========================================
    # 7. Submission
    # ==========================================
    threshold = 0.41003785424660755
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Inference on Test Set
        _, test_preds, _ = eval_fn(model, test_loader, device)

        # Load Sample Submission
        submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Verify shapes
        if test_preds.shape[0] == len(submission):
            submission[Config.TARGET_COLS] = test_preds

            # Save Submission
            submission.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
        else:
            print(
                f"Error: Test predictions shape {test_preds.shape} does not match submission shape {submission.shape}"
            )
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission generation."
        )


if __name__ == "__main__":
    run()
