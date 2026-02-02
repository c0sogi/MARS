import os
import sys
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score
from scipy.stats import pointbiserialr

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.trainer import Trainer


def get_audio_duration(filepath):
    """Helper to get duration of a wav file."""
    try:
        f = sf.SoundFile(filepath)
        return float(len(f) / f.samplerate)
    except Exception:
        return 0.0


def run_validation_inference(trainer):
    """
    Runs inference on the validation set using the trainer's model and processor.
    Returns:
        preds (np.array): Predicted class indices.
        targets (np.array): True class indices.
    """
    trainer.model.eval()
    trainer.processor.eval()

    all_preds = []
    all_targets = []

    device = trainer.device

    print("Running inference on validation set...")
    with torch.no_grad():
        for waveforms, labels in trainer.val_loader:
            waveforms = waveforms.to(device, non_blocking=True)

            # Process audio (Spectrograms, etc.)
            features = trainer.processor(waveforms)

            # Forward pass
            logits = trainer.model(features)

            # Get predictions
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_targets = labels.numpy()

            all_preds.append(batch_preds)
            all_targets.append(batch_targets)

    return np.concatenate(all_preds), np.concatenate(all_targets)


def main():
    # 1. Configuration Override
    # We increase epochs to 25 as the smaller native resolution inputs allow for faster iteration.
    Config.EPOCHS = 25

    # Ensure setup
    Config.setup()
    set_seed(Config.SEED)

    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Initialize Trainer
    # This loads data (cached) and initializes model/optimizer
    trainer = Trainer()

    # 3. Train
    trainer.fit()

    # 4. Load Best Model for Evaluation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        state_dict = torch.load(best_model_path, map_location=trainer.device)
        trainer.model.load_state_dict(state_dict)
    else:
        print("Warning: Best model not found. Using current model state.")

    # 5. Validation & Metric Calculation
    val_preds, val_targets = run_validation_inference(trainer)

    final_metric = accuracy_score(val_targets, val_preds)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Load validation metadata to get file paths
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment (DataLoader is sequential with shuffle=False)
    if len(val_df) != len(val_preds):
        print("Warning: Validation DataFrame length mismatch with predictions.")
        # Truncate to match if necessary (though they should match)
        min_len = min(len(val_df), len(val_preds))
        val_df = val_df.iloc[:min_len]
        val_preds = val_preds[:min_len]
        val_targets = val_targets[:min_len]

    # Calculate Error Vector (1 if wrong, 0 if correct)
    errors = (val_preds != val_targets).astype(int)

    # Extract Feature: Duration
    # We read this from disk. Since we need to be fast, we can use sf.info or just assume
    # most are 1s. However, the task asks for correlation with input features.
    # Let's compute it for a subset if it's too slow, but for ~11k files it should be fast enough.
    print("Extracting audio durations for correlation analysis...")
    durations = []
    for idx, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_ROOT, row["filepath"])
        durations.append(get_audio_duration(full_path))

    durations = np.array(durations)

    # Correlation Analysis
    # Point-biserial correlation is used when one variable is binary (error) and other is continuous (duration)
    # If constant duration, correlation is NaN.
    if np.std(durations) > 0:
        corr, p_value = pointbiserialr(errors, durations)
        print(
            f"Correlation between Error and Audio Duration: {corr:.4f} (p-value: {p_value:.4f})"
        )

        if abs(corr) > 0.1:
            print("-> Significant correlation detected. Duration impacts performance.")
        else:
            print(
                "-> Low correlation. Duration likely does not significantly impact performance."
            )
    else:
        print("Duration is constant across validation set. Correlation undefined.")

    # 7. Submission
    TARGET_THRESHOLD = 0.9836646499567848

    if final_metric > TARGET_THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({TARGET_THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({TARGET_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
