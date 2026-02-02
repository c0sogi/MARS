import os
import sys
import warnings
import torch
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import SiameseDualEncoder
from library.engine import (
    get_optimizer_params,
    get_scheduler,
    train_one_epoch,
    validate,
    predict,
)

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    # Using load_cached_data=True to utilize preprocessed parquet files if available
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing Siamese Dual-Encoder Model...")
    model = SiameseDualEncoder()
    model.to(device)

    # 4. Optimization Setup
    # Differential Learning Rates: Higher for head, lower for backbone
    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params)

    # Phantom Scheduling Strategy
    # We schedule decay for Config.SCHEDULER_EPOCHS (7) but only train for Config.EPOCHS (3)
    steps_per_epoch = len(train_loader)
    total_scheduler_steps = steps_per_epoch * Config.SCHEDULER_EPOCHS
    scheduler = get_scheduler(optimizer, total_scheduler_steps)

    # 5. Training Loop
    best_score = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train one epoch
        # The engine handles Head Warmup (freezing backbone in epoch 0) internally
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch
        )

        # Validate
        val_loss, val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Score: {val_score:.6f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (Score: {best_score:.6f})")

    # 6. Final Validation Metric
    # Printing full precision as required
    print(f"Final Validation Metric: {best_score}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Load best model for analysis
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    # Get predictions on validation set
    # val_loader has shuffle=False, so order matches metadata/val.csv
    val_preds = predict(model, val_loader, device)

    # Load validation metadata to get targets and features
    val_df = pd.read_csv(Config.VAL_PATH)
    val_targets = val_df[Config.TARGET_COLS].values

    # Calculate Mean Absolute Error per sample
    # Shape: (N_samples, 30) -> (N_samples,)
    abs_errors = np.abs(val_preds - val_targets)
    mean_abs_error = np.mean(abs_errors, axis=1)

    # Compute feature lengths for correlation analysis
    val_df["title_len"] = val_df["question_title"].fillna("").astype(str).str.len()
    val_df["body_len"] = val_df["question_body"].fillna("").astype(str).str.len()
    val_df["answer_len"] = val_df["answer"].fillna("").astype(str).str.len()

    features_to_analyze = ["title_len", "body_len", "answer_len"]

    print("Correlation between Error Magnitude and Input Features:")
    for feature in features_to_analyze:
        if feature in val_df.columns:
            corr, _ = spearmanr(val_df[feature], mean_abs_error)
            print(f"  {feature}: {corr:.6f}")

    # 8. Submission Generation
    THRESHOLD = 0.4118214482019393

    if best_score > THRESHOLD:
        print(
            f"\nValidation score ({best_score:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = predict(model, test_loader, device)

        # Load test metadata to get qa_ids
        test_df = pd.read_csv(Config.TEST_PATH)

        # Create submission DataFrame
        submission_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
        submission_df.insert(0, "qa_id", test_df["qa_id"])

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score ({best_score:.6f}) is below threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
