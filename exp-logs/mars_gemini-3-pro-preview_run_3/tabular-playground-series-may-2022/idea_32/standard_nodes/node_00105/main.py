import sys
import os
import numpy as np
import pandas as pd
import torch

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.data_processing import get_dataloaders
from library.model import DSPFE
from library.train_eval import train_model, validate, predict_submission


def perform_failure_analysis(model, val_loader, device, feature_names):
    """
    Analyzes model errors on the validation set.
    Calculates the correlation between absolute error and continuous features.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_cont = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x_cat, x_cont, targets in val_loader:
            x_cat = x_cat.to(device)
            # Move cont to GPU for model, keep copy or move back for analysis
            x_cont_gpu = x_cont.to(device)

            main_outs, _ = model(x_cat, x_cont_gpu)

            # Average probabilities
            probs = [torch.sigmoid(out) for out in main_outs]
            avg_prob = torch.stack(probs).mean(dim=0)

            all_preds.append(avg_prob.cpu().numpy())
            all_targets.append(targets.numpy())
            all_cont.append(x_cont.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_cont = np.concatenate(all_cont, axis=0)

    # Calculate absolute error
    errors = np.abs(all_targets - all_preds)

    # Create DataFrame for analysis
    # Note: feature_names corresponds to cont_cols
    df_analysis = pd.DataFrame(all_cont, columns=feature_names)
    df_analysis["error"] = errors

    # Compute correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("Top 5 Continuous Features correlated with Prediction Error:")
    print(correlations.head(5))
    return correlations


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Update Submission Path to meet specific task requirement
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # 2. Data Loading
    # We use the full dataset to ensure we meet the high AUC threshold.
    # The A100 GPU is sufficient to handle this volume quickly.
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, config=Config
    )

    # 3. Model Initialization
    print("Initializing DSPFE Model...")
    model = DSPFE(
        vocab_sizes=metadata["vocab_sizes"],
        num_cont=len(metadata["cont_cols"]),
        stream_configs=Config.STREAMS_CONFIG,
        embed_dim=Config.EMBEDDING_DIM,
    )

    # 4. Training
    # train_model handles the loop, optimizer, scheduler, and saving best_model.pth
    train_model(model, train_loader, val_loader, Config)

    # 5. Validation Assessment
    # Load the best model saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model file not found. Using current model weights.")

    model.to(Config.DEVICE)

    # Compute Final Metric
    final_auc = validate(model, val_loader, Config.DEVICE)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    perform_failure_analysis(model, val_loader, Config.DEVICE, metadata["cont_cols"])

    # 6. Submission
    # Threshold check
    THRESHOLD = 0.9975746465492954

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Load test IDs from metadata file
        # Config.TEST_PATH points to ./metadata/test.csv which has 'id' column
        test_df = pd.read_csv(Config.TEST_PATH)
        test_ids = test_df["id"].values

        predict_submission(model, test_loader, test_ids, Config)
    else:
        print(
            f"\nValidation AUC ({final_auc}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
