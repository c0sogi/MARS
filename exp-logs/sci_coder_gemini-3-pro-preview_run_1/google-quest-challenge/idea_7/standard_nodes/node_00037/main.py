import os
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup

# Import from provided library files
from library.config import Config, seed_everything
from library.data import get_dataloaders, get_data
from library.model import MultiTaskDualEncoder, get_optimizer_params
from library.engine import train_one_epoch, validate, predict
from library.utils import compute_spearman_metric


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Setup device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load DataLoaders
    # We use debug=False to train on the full metadata split (approx 4400 samples)
    # This is small enough to run quickly (within minutes on A100)
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing model...")
    model = MultiTaskDualEncoder()
    model.to(device)

    # ==========================================
    # 4. Optimizer & Scheduler
    # ==========================================
    optimizer_params = get_optimizer_params(
        model, Config.LR_BACKBONE, Config.LR_HEAD, Config.WEIGHT_DECAY
    )
    optimizer = torch.optim.AdamW(optimizer_params)

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    scaler = GradScaler()

    # ==========================================
    # 5. Training Loop
    # ==========================================
    best_score = -1.0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, scaler
        )

        # Validate
        val_score = validate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Val Spearman: {val_score:.10f}")

        # Save Best Model
        if val_score > best_score:
            print(
                f"Score Improved ({best_score:.5f} -> {val_score:.5f}). Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"Training Complete. Best Val Score: {best_score:.10f}")

    # ==========================================
    # 6. Final Evaluation & Failure Analysis
    # ==========================================
    print("\n--- Starting Final Evaluation & Failure Analysis ---")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Generate predictions on validation set
    val_preds = predict(model, val_loader, device)

    # Get ground truth targets
    # Note: QADataset stores targets in self.targets
    val_targets = val_loader.dataset.targets

    # Compute Final Metric
    final_metric = compute_spearman_metric(val_targets, val_preds)
    # Requirement: Print full precision
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # 1. Calculate Mean Absolute Error per sample
    abs_errors = np.abs(val_targets - val_preds)
    mean_abs_error = np.mean(abs_errors, axis=1)

    # 2. Load validation dataframe to get input features
    # We use get_data to retrieve the dataframe (cached)
    _, val_df, _ = get_data(load_cached_data=True)

    # Ensure alignment (in case of any data loading discrepancies)
    if len(val_df) != len(mean_abs_error):
        print(
            f"Warning: Shape mismatch for failure analysis. DF: {len(val_df)}, Errors: {len(mean_abs_error)}"
        )
        val_df = val_df.iloc[: len(mean_abs_error)]

    # 3. Extract features for correlation
    # Feature 1: Question Length (Title + Body)
    val_df["q_len"] = (
        val_df["question_title"].fillna("").str.len()
        + val_df["question_body"].fillna("").str.len()
    )
    # Feature 2: Answer Length
    val_df["a_len"] = val_df["answer"].fillna("").str.len()

    # 4. Compute Correlations
    corr_q = val_df["q_len"].corr(pd.Series(mean_abs_error))
    corr_a = val_df["a_len"].corr(pd.Series(mean_abs_error))

    print(f"Correlation between Error and Question Length: {corr_q}")
    print(f"Correlation between Error and Answer Length: {corr_a}")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    THRESHOLD = 0.40802662717842303

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = predict(model, test_loader, device)

        # Load Test DataFrame for IDs
        test_df = pd.read_csv(Config.TEST_PATH)

        # Create Submission DataFrame
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
