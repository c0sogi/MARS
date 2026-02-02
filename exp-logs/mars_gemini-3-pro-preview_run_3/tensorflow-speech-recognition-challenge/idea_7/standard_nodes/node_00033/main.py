import os
import sys
import pandas as pd
import numpy as np
import torch
import soundfile as sf
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

from library.config import Config
from library.utils import set_seed
from library.training_loop import train_model, predict_submission
from library.network import MR_SK_CRNN
from library.dataset import get_dataloaders


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Train
    # We use the full dataset (debug=False) to achieve the high accuracy target.
    # Caching handles the preprocessing efficiently.
    print("Starting training pipeline...")
    train_model(debug=False)

    # 3. Validation & Failure Analysis
    print("Starting evaluation...")
    device = Config.DEVICE

    # Load the best model saved during training
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Training may have failed."
        )
        return

    model = MR_SK_CRNN().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Get validation data
    # Note: get_dataloaders returns (train_loader, val_loader)
    # We rely on load_cached_data=True to use the features generated during training
    _, val_loader = get_dataloaders(debug=False, load_cached_data=True)

    all_preds = []
    all_labels = []
    all_probs = []

    # Inference on Validation Set
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate Metric
    val_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Final Validation Metric: {val_accuracy}")

    # Failure Analysis
    print("Performing failure analysis...")

    # Load metadata
    df_val = pd.read_csv(Config.VAL_METADATA)

    # Check alignment (assuming 1:1 as per dataset analysis logic)
    if len(df_val) != len(all_labels):
        print(
            f"Warning: Metadata length ({len(df_val)}) matches predictions ({len(all_labels)}) mismatch. Truncating to minimum."
        )
        min_len = min(len(df_val), len(all_labels))
        df_val = df_val.iloc[:min_len]
        all_labels = all_labels[:min_len]
        all_probs = all_probs[:min_len]

    # Calculate Error Magnitude (1 - prob of correct class)
    # Using numpy indexing for speed
    rows = np.arange(len(all_labels))
    correct_class_probs = all_probs[rows, all_labels]
    error_magnitudes = 1.0 - correct_class_probs

    # Get Durations
    # We read durations from files.
    durations = []
    for filepath in df_val["filepath"]:
        full_path = os.path.join(Config.INPUT_DIR, filepath)
        try:
            info = sf.info(full_path)
            durations.append(info.duration)
        except:
            durations.append(0.0)

    df_val["duration"] = durations
    df_val["error_magnitude"] = error_magnitudes
    df_val["label_id"] = all_labels

    # Correlations
    # 1. Error Magnitude vs Duration
    corr_dur, _ = pearsonr(df_val["duration"], df_val["error_magnitude"])
    print(f"Correlation (Error Magnitude vs Duration): {corr_dur:.6f}")

    # 2. Error Magnitude vs Label ID (Categorical proxy)
    corr_label, _ = pearsonr(df_val["label_id"], df_val["error_magnitude"])
    print(f"Correlation (Error Magnitude vs Label ID): {corr_label:.6f}")

    # 4. Submission Logic
    # Threshold from prompt
    THRESHOLD = 0.9832324978392394

    if val_accuracy > THRESHOLD:
        print(
            f"Validation accuracy {val_accuracy} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        predict_submission(debug=False)
    else:
        print(
            f"Validation accuracy {val_accuracy} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
