import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import provided libraries
from library.config import Config
from library.data_loader import get_dataloaders
from library.trainer import Trainer


# ------------------------------------------------------------------------------
# 1. Setup and Reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(Config.SEED)

# Override Config for Fast Baseline execution
# The dataset is small (~400 samples), so 15 epochs is appropriate.
Config.NUM_EPOCHS = 15


# ------------------------------------------------------------------------------
# 2. Orchestration
# ------------------------------------------------------------------------------
def main():
    print("Starting Fast Baseline Run...")

    # Initialize Trainer (sets up model, optimizer, criterion)
    trainer = Trainer()

    # Get DataLoaders
    # We use the full dataset as it is small enough for the time limit.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    print(f"Training on {len(train_loader.dataset)} samples.")
    print(f"Validating on {len(val_loader.dataset)} samples.")

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    best_val_auc = 0.0

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss, train_auc = trainer.train_one_epoch(train_loader)

        # Validate
        val_loss, val_auc = trainer.validate(val_loader)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}"
        )

        # Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(trainer.model.state_dict(), trainer.model_path)
            # Reset patience handled manually here if needed, but we just run fixed epochs for baseline

    # --------------------------------------------------------------------------
    # Final Evaluation & Metric
    # --------------------------------------------------------------------------
    # Load best model for final evaluation
    if os.path.exists(trainer.model_path):
        trainer.model.load_state_dict(
            torch.load(trainer.model_path, map_location=trainer.device)
        )

    # Re-run validation to get predictions for failure analysis and final metric
    trainer.model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device).unsqueeze(1)

            logits = trainer.model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    # Compute Final Metric
    if len(np.unique(all_targets)) < 2:
        final_auc = 0.5
    else:
        final_auc = roc_auc_score(all_targets, all_preds)

    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate errors
    errors = np.abs(all_targets - all_preds)

    # We need to correlate errors with a metadata feature.
    # Let's count the number of FLAIR slices for each validation subject.
    # We need to access the validation metadata again.
    df_val = pd.read_csv(Config.VAL_METADATA)

    flair_counts = []
    for idx, row in df_val.iterrows():
        flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        if os.path.exists(flair_path):
            # Fast count of files
            try:
                count = len(
                    [name for name in os.listdir(flair_path) if name.endswith(".dcm")]
                )
            except:
                count = 0
        else:
            count = 0
        flair_counts.append(count)

    flair_counts = np.array(flair_counts)

    # Ensure alignment (DataLoader preserves order if shuffle=False, which it is for val)
    if len(flair_counts) == len(errors):
        corr, _ = pearsonr(errors, flair_counts)
        print(f"Correlation between Error Magnitude and FLAIR Slice Count: {corr:.4f}")

        # Insight
        if abs(corr) > 0.1:
            print("Insight: Model performance is correlated with scan depth.")
        else:
            print("Insight: Model performance is relatively independent of scan depth.")
    else:
        print("Warning: Mismatch in validation set size for failure analysis.")

    # --------------------------------------------------------------------------
    # Conditional Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.6254545454545455

    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(test_loader)
    else:
        print(
            f"\nMetric ({final_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
