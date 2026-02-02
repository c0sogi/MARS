import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import provided library functions
from library.utils import seed_everything, get_device
from library.data import load_dataset_split
from library.model import EffNet25D
from library.train import run_training
from library.inference import predict_submission


def main():
    # 1. Configuration and Setup
    seed_everything(42)
    device = get_device()

    # Constants
    WORKING_DIR = "./working/idea_37"
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    METADATA_VAL_PATH = "./metadata/val.parquet"
    SUBMISSION_PATH = "./submission/submission.csv"
    THRESHOLD_METRIC = 0.6978181818181817

    print(f"Running on device: {device}")

    # 2. Train the Model
    # We use 15 epochs which is sufficient for this small dataset size (~400 samples)
    # to reach convergence without exceeding time limits.
    print("Starting training...")
    _ = run_training(
        epochs=15,
        batch_size=16,
        learning_rate=1e-4,
        patience=5,
        save_dir=WORKING_DIR,
        load_cached_data=True,
    )

    # 3. Validation & Evaluation
    print("Loading validation data for evaluation...")
    X_val, y_val, ids_val = load_dataset_split("val", load_cached_data=True)

    # Load the best model
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found.")
        return

    model = EffNet25D(model_name="efficientnet_b0", pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Inference on Validation Set
    val_probs = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(y_val), batch_size):
            # Prepare batch
            x_batch = torch.tensor(X_val[i : i + batch_size]).to(device)

            # Forward
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            val_probs.extend(probs)

    val_probs = np.array(val_probs)

    # Calculate Metric
    # Handle edge case if y_val contains only one class (unlikely given stratification)
    try:
        final_metric = roc_auc_score(y_val, val_probs)
    except ValueError:
        final_metric = 0.5

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - val_probs)

    # Load metadata to extract features for correlation
    if os.path.exists(METADATA_VAL_PATH):
        val_df = pd.read_parquet(METADATA_VAL_PATH)

        # Ensure the dataframe aligns with the loaded numpy arrays
        # load_dataset_split iterates through the dataframe sequentially.

        # Extract features: Slice counts per modality
        # Note: Paths are lists in the parquet file
        features = {
            "flair_count": val_df["flair_paths"].apply(len).values,
            "t1w_count": val_df["t1w_paths"].apply(len).values,
            "t1wce_count": val_df["t1wce_paths"].apply(len).values,
            "t2w_count": val_df["t2w_paths"].apply(len).values,
            "target_value": y_val,
        }

        print("Correlation between Error Magnitude and Metadata Features:")
        for feat_name, feat_values in features.items():
            # Check for constant values to avoid warnings
            if np.std(feat_values) == 0 or np.std(errors) == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(feat_values, errors)
            print(f" - {feat_name}: {corr:.6f}")
    else:
        print(
            "Warning: Validation metadata not found, skipping feature correlation analysis."
        )

    # 5. Submission
    if final_metric > THRESHOLD_METRIC:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD_METRIC})."
        )
        print("Generating submission file...")
        predict_submission(
            model_path=MODEL_PATH,
            save_path=SUBMISSION_PATH,
            batch_size=16,
            load_cached_data=True,
            device=device,
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({THRESHOLD_METRIC})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
