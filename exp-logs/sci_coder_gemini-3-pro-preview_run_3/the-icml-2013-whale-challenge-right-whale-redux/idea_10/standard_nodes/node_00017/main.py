import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from library.config import Config
from library.utils import set_seed, calculate_auc
from library.train import run_training, generate_submission
from library.models import WhaleClassifier
from library.dataset import get_dataloaders


def analyze_failures(val_df, predictions, targets):
    """
    Performs failure analysis by correlating error magnitude with input features.
    """
    print("\n=== Failure Analysis ===")

    # Add predictions and targets to dataframe
    val_df["probability"] = predictions
    val_df["label"] = targets

    # Calculate absolute error
    val_df["error"] = (val_df["label"] - val_df["probability"]).abs()

    # Extract audio features
    durations = []
    mean_amps = []
    std_amps = []

    print("Extracting audio features for validation set...")
    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        try:
            # Read audio
            data, sr = sf.read(full_path)

            # Handle multi-channel by averaging
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            durations.append(len(data) / sr)
            mean_amps.append(np.mean(np.abs(data)))
            std_amps.append(np.std(data))
        except Exception as e:
            # Fallback for read errors
            durations.append(0)
            mean_amps.append(0)
            std_amps.append(0)

    val_df["duration"] = durations
    val_df["mean_amp"] = mean_amps
    val_df["std_amp"] = std_amps

    # Calculate Correlations
    print("Correlation between Error Magnitude and Input Features:")
    features = ["duration", "mean_amp", "std_amp"]

    for feat in features:
        # Check if feature has variance to avoid division by zero/NaN
        if val_df[feat].std() > 1e-9:
            corr = np.corrcoef(val_df["error"], val_df[feat])[0, 1]
            print(f"  {feat}: {corr:.6f}")
        else:
            print(f"  {feat}: NaN (No variance in feature)")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)

    # Override Config for Fast Baseline
    # Increasing epochs to 15 to ensure convergence with larger backbone
    Config.EPOCHS = 15

    print(f"Starting execution with EPOCHS={Config.EPOCHS}")

    # 2. Training
    # run_training handles the training loop, validation, and saves the best model
    run_training(load_cached_data=True)

    # 3. Validation Assessment
    print("\n=== Final Validation Assessment ===")
    device = torch.device(Config.DEVICE)

    # Load Best Model
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Best model not found at {Config.BEST_MODEL_PATH}")
        sys.exit(1)

    model = WhaleClassifier().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Get Validation Data
    # We ignore the train loader here
    _, val_loader = get_dataloaders(load_cached_data=True)

    all_preds = []
    all_targets = []

    # Inference
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(labels.numpy().flatten())

    # Calculate Metric
    val_auc = calculate_auc(all_targets, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    # Load validation metadata to map back to files
    val_df = pd.read_csv(Config.VAL_CSV)
    analyze_failures(val_df, all_preds, all_targets)

    # 5. Submission Generation
    THRESHOLD = 0.9960914834372254

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation AUC ({val_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
