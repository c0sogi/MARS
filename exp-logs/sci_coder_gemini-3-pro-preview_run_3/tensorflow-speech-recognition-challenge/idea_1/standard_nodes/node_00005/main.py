import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
import library.dataset as dataset
import library.model as model_lib
import library.trainer as trainer
import library.utils as utils


def get_audio_features(filepath):
    """
    Extracts basic features (duration, rms energy) from an audio file
    for failure analysis.
    """
    full_path = os.path.join(config.INPUT_DIR, filepath)
    try:
        data, sr = sf.read(full_path)
        duration = len(data) / sr
        # RMS Energy
        rms = np.sqrt(np.mean(data**2)) if len(data) > 0 else 0.0
        return duration, rms
    except Exception:
        return 0.0, 0.0


def run_failure_analysis(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error and input features.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    all_preds = []
    all_labels = []

    # 1. Collect Predictions
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 2. Compute Metric
    acc = accuracy_score(all_labels, all_preds)
    print(f"Final Validation Metric: {acc}")

    # 3. Analyze Errors
    # Ensure dataframe aligns with loader (dataset.py resets index for val)
    analysis_df = val_df.copy().reset_index(drop=True)
    analysis_df["pred_idx"] = all_preds
    analysis_df["true_idx"] = all_labels

    # Binary Error: 1 if wrong, 0 if correct
    analysis_df["is_error"] = (
        analysis_df["pred_idx"] != analysis_df["true_idx"]
    ).astype(int)

    # Extract features for correlation analysis
    # We extract features for a subset if it's too large, but val is ~11k, which is manageable.
    print("Extracting audio features for correlation analysis...")
    durations = []
    energies = []

    for idx, row in analysis_df.iterrows():
        dur, nrg = get_audio_features(row["filepath"])
        durations.append(dur)
        energies.append(nrg)

    analysis_df["duration"] = durations
    analysis_df["energy"] = energies

    # Calculate Correlations
    # We look for correlation between 'is_error' and features
    corr_duration = analysis_df["is_error"].corr(analysis_df["duration"])
    corr_energy = analysis_df["is_error"].corr(analysis_df["energy"])

    print(f"Correlation (Error vs Duration): {corr_duration:.4f}")
    print(f"Correlation (Error vs Signal Energy): {corr_energy:.4f}")

    if abs(corr_duration) > 0.1:
        print("-> Note: Model performance seems sensitive to audio duration.")
    if abs(corr_energy) > 0.1:
        print("-> Note: Model performance seems sensitive to signal volume.")


def main():
    # 1. Configuration and Setup
    config.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train Model
    # We use the provided trainer function.
    # Increased patience to 8 because ResNet converges slower than simple CNN (Cite solution_lesson_node_00003)
    print("Starting Training...")
    model = trainer.train_model(epochs=20, patience=8)

    # 3. Validation & Failure Analysis
    # We need to reload the validation set to perform custom analysis
    val_csv = os.path.join(config.METADATA_DIR, "val.csv")
    df_val = pd.read_csv(val_csv)

    val_dataset = dataset.SpeechCommandsDataset(df_val, phase="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Calculate metric for conditional submission
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    final_acc = accuracy_score(all_labels, all_preds)

    run_failure_analysis(model, val_loader, df_val, device)

    # 4. Generate Submission
    # Only submit if accuracy exceeds threshold
    if final_acc > 0.9527:
        print(f"Validation Accuracy {final_acc:.4f} > 0.9527. Generating submission...")
        trainer.generate_submission(model)
    else:
        print(f"Validation Accuracy {final_acc:.4f} <= 0.9527. Skipping submission.")


if __name__ == "__main__":
    main()
