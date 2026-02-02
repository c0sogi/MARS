import os
import sys
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.dataset import SpeechCommandsDataset
from library.model import SpectroCNN
from library.trainer import run_training, generate_submission


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 1. Train the model and generate submission
    # The run_training function handles the entire training loop, validation monitoring,
    # best model saving, and final submission generation on the test set.
    print("Executing Training Pipeline...")
    run_training(load_cached_data=True)

    # 2. Final Validation Evaluation
    # We need to reload the best model and the validation set to compute the final metric
    # and perform failure analysis.
    print("\nExecuting Validation Evaluation...")

    device = Config.DEVICE

    # Load Validation Dataset
    val_dataset = SpeechCommandsDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = SpectroCNN(num_classes=Config.NUM_CLASSES)
    model.to(device)

    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using initialized weights.")

    model.eval()

    all_preds = []
    all_labels = []

    # Inference loop
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate Accuracy
    accuracy = np.mean(all_preds == all_labels)

    # Print Required Metric
    print(f"Final Validation Metric: {accuracy}")

    # Conditional Submission Generation
    TARGET_METRIC = 0.9523411371237458
    if accuracy > TARGET_METRIC:
        print(
            f"Validation metric ({accuracy:.6f}) > target ({TARGET_METRIC:.6f}). Generating submission..."
        )
        generate_submission(model, device, load_cached_data=True)
    else:
        print(
            f"Validation metric ({accuracy:.6f}) <= target ({TARGET_METRIC:.6f}). Skipping submission."
        )

    # 3. Failure Analysis
    print("\nExecuting Failure Analysis...")

    # Create a DataFrame for analysis
    # We use the validation dataset metadata
    df_analysis = val_dataset.df.copy()

    # Ensure alignment
    if len(df_analysis) != len(all_preds):
        print("Error: Mismatch between validation set size and predictions.")
    else:
        df_analysis["predicted"] = all_preds
        df_analysis["actual"] = all_labels
        # Error: 1 if incorrect, 0 if correct
        df_analysis["is_error"] = (
            df_analysis["predicted"] != df_analysis["actual"]
        ).astype(int)

        # To analyze correlations, we need signal features (Duration, RMS).
        # We sample a subset to keep runtime low (e.g., 2000 samples).
        sample_size = min(2000, len(df_analysis))
        df_sample = df_analysis.sample(n=sample_size, random_state=Config.SEED).copy()

        durations = []
        rms_values = []

        print(f"Extracting features from {sample_size} validation files...")

        for idx, row in df_sample.iterrows():
            filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])
            try:
                # Load audio
                wav, sr = torchaudio.load(filepath)

                # Duration (seconds)
                dur = wav.shape[1] / sr
                durations.append(dur)

                # RMS (Root Mean Square) - measure of loudness
                rms = torch.sqrt(torch.mean(wav**2)).item()
                rms_values.append(rms)

            except Exception:
                durations.append(0.0)
                rms_values.append(0.0)

        df_sample["duration"] = durations
        df_sample["rms"] = rms_values

        # Calculate Correlations
        # We check if 'is_error' correlates with 'duration' or 'rms'
        # Using Pearson correlation (point-biserial equivalent for binary)

        # Filter out invalid reads
        df_clean = df_sample[df_sample["duration"] > 0]

        if len(df_clean) > 10 and df_clean["is_error"].std() > 0:
            corr_dur = np.corrcoef(df_clean["is_error"], df_clean["duration"])[0, 1]
            corr_rms = np.corrcoef(df_clean["is_error"], df_clean["rms"])[0, 1]

            print("Correlation between Error Magnitude and Input Features:")
            print(f"  Error vs Duration: {corr_dur:.6f}")
            print(f"  Error vs RMS (Loudness): {corr_rms:.6f}")

            if abs(corr_dur) > 0.1:
                print("  Observation: Duration seems to impact error rate.")
            if abs(corr_rms) > 0.1:
                print("  Observation: Loudness seems to impact error rate.")
        else:
            print("Insufficient variance in errors or data to compute correlation.")


if __name__ == "__main__":
    main()
