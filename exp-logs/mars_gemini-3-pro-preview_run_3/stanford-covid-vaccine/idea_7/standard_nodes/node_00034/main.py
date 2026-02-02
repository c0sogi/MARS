import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, mcrmse_metric, save_submission
from library.dataset import get_dataloaders
from library.model import SpatiallyAugmentedBiGRU
from library.train import Trainer


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    # Initialize system paths and seeds
    Config.setup_system()
    set_seed(Config.SEED)

    # Define paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets...")
    # load_cached_data=True allows using preprocessed artifacts if available
    loaders = get_dataloaders(load_cached_data=True)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # ==========================================
    # 3. Model Initialization & Training
    # ==========================================
    print("Initializing model and trainer...")
    device = torch.device(Config.DEVICE)
    model = SpatiallyAugmentedBiGRU().to(device)

    trainer = Trainer(model, train_loader, val_loader, device)

    print("Starting training...")
    trainer.fit(epochs=Config.EPOCHS)

    # ==========================================
    # 4. Validation Evaluation
    # ==========================================
    print("Evaluating best model on validation set...")
    # Load the best model state saved during training
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_val_preds = []
    all_val_targets = []

    # Inference loop - No gradients for speed
    with torch.no_grad():
        for X, y, _ in val_loader:
            X = X.to(device)
            preds = model(X)

            # Slice predictions to match the scored sequence length (68)
            preds_sliced = preds[:, : Config.SEQ_SCORED, :]

            all_val_preds.append(preds_sliced.cpu().numpy())
            all_val_targets.append(y.numpy())

    # Concatenate batches
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_targets = np.concatenate(all_val_targets, axis=0)

    # Compute Metric
    val_metric = mcrmse_metric(all_val_targets, all_val_preds)
    print(f"Final Validation Metric: {val_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")
    # Calculate RMSE per sample (averaged across all 5 targets for general error analysis)
    # Shape: (N_samples, Seq_Scored, Targets)
    squared_errors = (all_val_preds - all_val_targets) ** 2
    # Mean over sequence and targets -> Scalar MSE per sample
    mse_per_sample = np.mean(squared_errors, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load validation metadata to correlate errors with features
    # Note: val_loader is not shuffled, so order matches the parquet file
    val_meta = pd.read_parquet(Config.VAL_METADATA_PATH)

    # Ensure lengths match (in case of any upstream filtering, though val shouldn't be filtered)
    min_len = min(len(val_meta), len(rmse_per_sample))
    val_meta = val_meta.iloc[:min_len]
    rmse_per_sample = rmse_per_sample[:min_len]

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "error": rmse_per_sample,
            "signal_to_noise": val_meta["signal_to_noise"].values,
            "SN_filter": val_meta["SN_filter"].values,
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Model Error and Metadata Features:")
    print(correlations)

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.7247761841173526

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for X, batch_ids in test_loader:
                X = X.to(device)
                preds = model(X)
                test_preds.append(preds.cpu().numpy())
                test_ids.extend(batch_ids)

        # Concatenate predictions
        test_preds = np.concatenate(test_preds, axis=0)

        # Save submission
        save_submission(test_preds, test_ids, save_path=SUBMISSION_FILE)
    else:
        print(
            f"\nValidation metric ({val_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
