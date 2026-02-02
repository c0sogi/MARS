import sys
import os
import torch
import numpy as np
import pandas as pd

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import HybridAttentionResFunnel
from library.trainer import Trainer


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Adjust Config for Fast Baseline Execution
    # A100 is available, so we can use a large batch size and reasonable epochs.
    # We limit epochs to ensure runtime compliance, relying on early stopping.
    Config.EPOCHS = 35
    Config.BATCH_SIZE = 4096

    print(f"Execution Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\nLoading Data...")
    # Load dataloaders (uses cached data if available)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Retrieve test_ids for submission (stored in the processed .npz file)
    if os.path.exists(Config.PROCESSED_DATA_PATH):
        data = np.load(Config.PROCESSED_DATA_PATH)
        test_ids = data["test_ids"]
    else:
        # Fallback if cache wasn't just created (unlikely given get_dataloaders ran)
        test_meta = pd.read_csv(Config.TEST_META_PATH)
        test_ids = test_meta["id"].values

    # --------------------------------------------------------------------------
    # 3. Model Initialization and Training
    # --------------------------------------------------------------------------
    print("\nInitializing Model...")
    model = HybridAttentionResFunnel()

    trainer = Trainer(model)

    print("\nStarting Training...")
    trainer.fit(train_loader, val_loader)

    # --------------------------------------------------------------------------
    # 4. Evaluation (Final Validation Metric)
    # --------------------------------------------------------------------------
    print("\nEvaluating Best Model on Validation Set...")

    # Load the best model weights saved during training
    if os.path.exists(Config.MODEL_PATH):
        trainer.model.load_state_dict(torch.load(Config.MODEL_PATH))

    # Compute metrics
    val_loss, val_auc = trainer.validate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # --------------------------------------------------------------------------
    # 5. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    trainer.model.eval()

    all_cont = []
    all_cat = []
    all_targets = []
    all_preds = []

    # Collect validation data and predictions
    with torch.no_grad():
        for batch in val_loader:
            cont = batch["cont_features"].to(Config.DEVICE)
            cat = batch["cat_sequence"].to(Config.DEVICE)
            target = batch["target"].to(Config.DEVICE)

            logits = trainer.model(cont, cat)
            probs = torch.sigmoid(logits).squeeze()

            all_cont.append(cont.cpu().numpy())
            all_cat.append(cat.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    # Concatenate
    X_cont = np.concatenate(all_cont, axis=0)
    X_cat = np.concatenate(all_cat, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Create Analysis DataFrame
    # Continuous features f_00 to f_30 (excluding f_27)
    feat_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
    df_analysis = pd.DataFrame(X_cont, columns=feat_cols)

    # Add Categorical components (indices)
    for i in range(10):
        df_analysis[f"f_27_char_{i}"] = X_cat[:, i]

    df_analysis["error"] = errors

    # Compute Correlation
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("Top 10 Features correlated with Prediction Error:")
    print(correlations.head(10))

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9963545814493672

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(test_loader, test_ids)
    else:
        print(
            f"\nValidation AUC ({val_auc}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
