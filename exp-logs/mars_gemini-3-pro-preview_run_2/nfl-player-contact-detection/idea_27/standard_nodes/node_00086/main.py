import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library components
import library.config
from library.config import WORKING_DIR, SEED
from library.utils import seed_everything, get_device
from library.train import train_model, get_vocab_sizes
from library.inference import optimize_threshold, generate_predictions, run_inference
from library.dataset import get_dataloaders
from library.model import SEARVN


def perform_failure_analysis(model_path):
    """
    Analyzes model errors on the validation set by correlating them with input features.
    """
    print("\n=== Failure Analysis ===")
    device = get_device()

    # 1. Load Validation Data (Raw Features)
    # The DataProcessor caches the processed dataframe which allows us to access readable features
    val_parquet_path = os.path.join(WORKING_DIR, "features_val.parquet")
    if not os.path.exists(val_parquet_path):
        print("Validation parquet not found. Skipping detailed failure analysis.")
        return

    df_val = pd.read_parquet(val_parquet_path)

    # 2. Get Model Predictions
    # We use the validation loader to ensure we get the exact tensors used by the model
    # shuffle=False ensures alignment with df_val
    _, val_loader = get_dataloaders(debug=False, load_cached_data=True)

    vocab_sizes = get_vocab_sizes()
    model = SEARVN(vocab_sizes=vocab_sizes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    probs, targets, _ = run_inference(model, val_loader, device)

    # 3. Calculate Error Magnitude
    # We look at the absolute difference between the binary target and the predicted probability
    errors = np.abs(targets - probs)

    # 4. Correlate with Physical Features
    # We extract key kinematic features from the dataframe to see what drives errors.
    # We use lag_0 (current time) features.

    # Calculate Euclidean Distance between players
    dx = df_val["x_position_lag_0_p1"] - df_val["x_position_lag_0_p2"]
    dy = df_val["y_position_lag_0_p1"] - df_val["y_position_lag_0_p2"]
    distance = np.sqrt(dx**2 + dy**2)

    # Extract Speeds
    speed_p1 = df_val["speed_lag_0_p1"]
    speed_p2 = df_val["speed_lag_0_p2"]

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "distance": distance.values,
            "speed_p1": speed_p1.values,
            "speed_p2": speed_p2.values,
        }
    )

    # Compute Pearson Correlation
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)


def main():
    # Set seeds for reproducibility
    seed_everything(SEED)

    print("Initializing End-to-End Pipeline...")

    # 1. Train the Model
    # We use the full dataset (debug=False) to maximize performance.
    # The library handles caching, so subsequent runs are faster.
    print("Step 1: Training Model...")
    best_model_path = train_model(debug=False, load_cached_data=True)

    # 2. Optimize Threshold
    # We find the threshold that maximizes MCC on the validation set.
    print("\nStep 2: Optimizing Threshold...")
    best_threshold, best_mcc = optimize_threshold(
        best_model_path, debug=False, load_cached_data=True
    )

    # 3. Report Metrics
    # Required format for evaluation
    print(f"Final Validation Metric: {best_mcc}")

    # 4. Failure Analysis
    # Identify systematic weaknesses
    perform_failure_analysis(best_model_path)

    # 5. Submission Generation
    # Strict conditional submission based on performance threshold
    TARGET_MCC = 0.6634847318478787

    if best_mcc > TARGET_MCC:
        print(f"\nPerformance Check: PASSED (MCC {best_mcc} > {TARGET_MCC})")
        print("Generating submission file...")
        generate_predictions(
            best_model_path, best_threshold, debug=False, load_cached_data=True
        )
    else:
        print(f"\nPerformance Check: FAILED (MCC {best_mcc} <= {TARGET_MCC})")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
