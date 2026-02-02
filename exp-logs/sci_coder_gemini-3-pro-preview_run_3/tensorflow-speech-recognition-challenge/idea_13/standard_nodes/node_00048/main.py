import os
import sys
import pandas as pd
import numpy as np
import torch
import soundfile as sf
from scipy.stats import pointbiserialr

# Import from provided library files
from library.config import (
    VAL_CSV,
    INPUT_ROOT,
    WORKING_DIR,
    SUBMISSION_PATH,
    MODEL_SAVE_PATH,
)
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.engine import Trainer


def perform_failure_analysis(trainer, val_loader, val_csv_path):
    """
    Analyzes model errors on the validation set.
    Calculates correlation between error status and audio duration.
    """
    print("\n=== Failure Analysis ===")

    # 1. Get Predictions and Targets
    trainer.model.eval()
    trainer.processor.eval()

    all_preds = []
    all_targets = []

    device = trainer.device

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Feature extraction (no augmentation)
            _, specs = trainer.processor(inputs)

            # Inference
            outputs = trainer.model(specs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 2. Calculate Error (1 = Wrong, 0 = Correct)
    errors = (all_preds != all_targets).astype(int)
    accuracy = 1.0 - np.mean(errors)
    print(f"Re-verified Validation Accuracy: {accuracy:.8f}")

    # 3. Load Metadata to get features (Duration)
    df_val = pd.read_csv(val_csv_path)

    # Ensure alignment: The val_loader is sequential (shuffle=False),
    # so indices should match df_val rows.
    if len(df_val) != len(errors):
        print(
            f"Warning: Metadata length ({len(df_val)}) matches predictions ({len(errors)}) mismatch."
        )
        return

    # Extract durations
    durations = []
    print("Extracting audio features for analysis...")
    for idx, row in df_val.iterrows():
        filepath = os.path.join(INPUT_ROOT, row["filepath"])
        try:
            info = sf.info(filepath)
            durations.append(info.duration)
        except:
            durations.append(0.0)

    durations = np.array(durations)

    # 4. Correlation Analysis
    # Point Biserial Correlation: Binary variable (Error) vs Continuous variable (Duration)
    # Handle case where all errors are 0 or 1 (correlation undefined)
    if np.std(errors) == 0:
        print("Model achieved 100% or 0% accuracy. Correlation undefined.")
    else:
        corr, p_value = pointbiserialr(errors, durations)
        print(
            f"Correlation between Error and Audio Duration: {corr:.6f} (p-value: {p_value:.6f})"
        )

        if abs(corr) > 0.1:
            print("-> Significant correlation detected. Duration impacts performance.")
        else:
            print("-> No significant correlation with duration.")


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using full dataset to maximize accuracy for the high threshold
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Trainer
    trainer = Trainer(learning_rate=1e-3, device=device)

    # 4. Train
    # We use 25 epochs to ensure convergence for the high metric requirement
    # The A100 is fast enough to handle this within the time limit.
    print("Starting training...")
    best_acc = trainer.fit(
        train_loader, val_loader, epochs=25, patience=5, save_path=MODEL_SAVE_PATH
    )

    # 5. Report Metric
    # MUST PRINT FULL PRECISION
    print(f"Final Validation Metric: {best_acc}")

    # 6. Failure Analysis
    perform_failure_analysis(trainer, val_loader, VAL_CSV)

    # 7. Submission
    THRESHOLD = 0.9832324978392394

    if best_acc > THRESHOLD:
        print(
            f"\nValidation metric ({best_acc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Ensure output directory exists
        submission_dir = os.path.dirname(SUBMISSION_PATH)
        if submission_dir and not os.path.exists(submission_dir):
            os.makedirs(submission_dir, exist_ok=True)

        trainer.generate_submission(
            test_loader, output_path=SUBMISSION_PATH, model_path=MODEL_SAVE_PATH
        )
    else:
        print(
            f"\nValidation metric ({best_acc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
