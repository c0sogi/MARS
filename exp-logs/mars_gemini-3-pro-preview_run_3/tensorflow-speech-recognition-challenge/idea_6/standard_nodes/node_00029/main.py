import os
import pandas as pd
import numpy as np
import torch
import soundfile as sf

from library.config import Config
from library.train import run_training, generate_submission
from library.dataset import get_dataloaders
from library.sk_resnet import get_model
from library.utils import get_device, set_seed


def main():
    # 1. Setup and Configuration
    Config.setup()
    set_seed(Config.SEED)

    # Optimize for fast baseline execution while maintaining performance.
    # 20 epochs is sufficient for SK-ResNet to converge on this dataset.
    Config.EPOCHS = 20

    print("=== Starting Runfile Execution ===")

    # 2. Run Training
    # This handles caching, training loop, and saving the best model.
    # We use the full dataset (max_samples=None) to ensure we can hit the high accuracy target.
    # run_training returns the test_loader which we might need for submission.
    test_loader = run_training(epochs=Config.EPOCHS)

    # 3. Load Validation Data for Analysis
    # We need the val_loader to perform detailed failure analysis and final metric calculation.
    # get_dataloaders is cached, so this is fast.
    print("\nLoading validation data for analysis...")
    _, val_loader, _ = get_dataloaders()

    # 4. Load Best Model
    device = get_device()
    model = get_model().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print("Loaded best model for evaluation.")
    else:
        print("Error: Model checkpoint not found.")
        return

    # 5. Final Validation & Failure Analysis
    model.eval()

    all_preds = []
    all_targets = []

    # Inference loop on Validation Set
    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Get predictions
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Accuracy
    correct_mask = all_preds == all_targets
    accuracy = np.mean(correct_mask)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {accuracy}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Load metadata to get features
    val_csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
    df_val = pd.read_csv(val_csv_path)

    # Ensure alignment (loader is not shuffled, so indices match)
    # Handle potential subsampling if DEBUG was somehow enabled in Config
    if len(df_val) != len(all_preds):
        print(
            "Warning: Mismatch between validation CSV and predictions length. Truncating for analysis."
        )
        df_val = df_val.iloc[: len(all_preds)]

    # Feature 1: Duration
    # Extract duration from files to check if length correlates with error
    print("Extracting audio features for correlation analysis...")
    durations = []
    for idx, row in df_val.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["filepath"])
        try:
            # sf.info is fast (reads header only)
            info = sf.info(full_path)
            durations.append(info.duration)
        except:
            durations.append(0.0)

    durations = np.array(durations)

    # Error Magnitude: 1 if error, 0 if correct
    error_magnitude = (~correct_mask).astype(int)

    # Calculate Correlation
    if np.std(error_magnitude) > 0 and np.std(durations) > 0:
        corr_duration = np.corrcoef(error_magnitude, durations)[0, 1]
        print(f"Correlation between Error and Audio Duration: {corr_duration:.4f}")
    else:
        print("Correlation could not be computed (constant values).")

    # Analyze Error by Class
    print("\nError Rate by Class:")
    df_val["pred"] = all_preds
    df_val["target"] = all_targets
    df_val["is_error"] = error_magnitude

    class_error = (
        df_val.groupby("label")["is_error"].mean().sort_values(ascending=False)
    )
    print(class_error)

    # 7. Conditional Submission
    THRESHOLD = 0.9771823681936042

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy ({accuracy}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(test_loader)
    else:
        print(
            f"\nValidation accuracy ({accuracy}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
