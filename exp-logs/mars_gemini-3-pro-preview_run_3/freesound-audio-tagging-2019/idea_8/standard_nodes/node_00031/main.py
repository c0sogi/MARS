import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.trainer import Trainer, generate_submission


def main():
    # 1. Configuration for Fast Baseline
    # We increase batch size to utilize A100 memory and reduce epochs to fit in the time limit
    # while maintaining enough iterations for convergence.
    Config.BATCH_SIZE = 128
    Config.EPOCHS = 8

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"Configuration: Batch Size={Config.BATCH_SIZE}, Epochs={Config.EPOCHS}")

    # 2. Data Loading
    # Load cached data to save preprocessing time
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Training
    trainer = Trainer(train_loader, val_loader, test_loader)
    trainer.fit()

    # 4. Validation Reporting
    # Print the exact metric required by the task
    print(f"Final Validation Metric: {trainer.best_score}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    analyze_failures(trainer, val_loader)

    # 6. Submission
    # Threshold defined in the task description
    THRESHOLD = 0.7117108825122853

    if trainer.best_score > THRESHOLD:
        print(f"\nScore {trainer.best_score} > {THRESHOLD}. Generating submission...")
        fnames, preds = trainer.predict_test()
        generate_submission(fnames, preds)
    else:
        print(f"\nScore {trainer.best_score} <= {THRESHOLD}. Submission skipped.")


def analyze_failures(trainer, val_loader):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    device = Config.DEVICE
    trainer.model.eval()

    all_preds = []
    all_targets = []

    # 1. Re-run inference on validation set to get raw predictions
    # (Trainer.validate returns score, not raw preds/targets)
    print("Computing validation predictions for analysis...")
    with torch.no_grad():
        for data, target, _ in val_loader:
            data = data.to(device)
            target = target.to(device)

            output = trainer.model(data)
            preds = torch.sigmoid(output)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 2. Calculate Error Magnitude
    # Using Mean Absolute Error per sample as the proxy for "error magnitude"
    # shape: (N_samples,)
    errors = np.mean(np.abs(preds - targets), axis=1)

    # 3. Load Metadata for Features
    # We need to ensure alignment. val_loader is shuffle=False, so it matches the CSV order.
    val_df = pd.read_csv(Config.VAL_META)

    if len(val_df) != len(errors):
        print(
            f"Warning: Metadata length ({len(val_df)}) != Prediction length ({len(errors)}). Skipping analysis."
        )
        return

    # Feature A: Number of Labels
    # We calculate this from the ground truth targets
    num_labels = targets.sum(axis=1)

    # Feature B: Audio Duration
    # We need to read the audio files to get duration.
    print("Extracting audio durations...")
    durations = []
    # Use a simple loop. Reading headers is fast.
    for idx, row in val_df.iterrows():
        filepath = os.path.join(Config.INPUT_DIR, row["filepath"])
        try:
            info = sf.info(filepath)
            durations.append(info.duration)
        except Exception as e:
            # Fallback for missing/corrupt files (though dataset should be clean)
            durations.append(0.0)

    durations = np.array(durations)

    # 4. Calculate Correlations
    # Use numpy for correlation
    # np.corrcoef returns the correlation matrix, we take the off-diagonal element
    corr_labels = np.corrcoef(errors, num_labels)[0, 1]
    corr_duration = np.corrcoef(errors, durations)[0, 1]

    print(f"Correlation (Error vs Num Labels): {corr_labels:.10f}")
    print(f"Correlation (Error vs Duration): {corr_duration:.10f}")


if __name__ == "__main__":
    main()
