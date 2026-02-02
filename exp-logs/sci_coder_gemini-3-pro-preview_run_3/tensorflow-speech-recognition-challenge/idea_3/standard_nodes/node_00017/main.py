import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import accuracy_score

# Import library modules
import library.config
from library.config import TRAIN_CONFIG, PATHS
from library.utils import set_seed
from library.trainer import Trainer


def patch_library_paths():
    """
    Patches the PATHS configuration in library.config.
    The provided metadata contains paths relative to './input' (e.g., 'train/audio/bed/123.wav').
    The provided Trainer uses PATHS['train_audio_dir'] which is './input/train/audio'.
    The Dataset class joins these, resulting in './input/train/audio/train/audio/bed/123.wav', which is invalid.
    We override the audio directories to './input' so the join results in correct paths.
    """
    library.config.PATHS["train_audio_dir"] = library.config.INPUT_DIR
    library.config.PATHS["test_audio_dir"] = library.config.INPUT_DIR


def get_audio_duration(filepath):
    """Helper to get duration of a wav file."""
    try:
        f = sf.info(filepath)
        return f.duration
    except Exception:
        return 0.0


def main():
    # 1. Configuration & Setup
    patch_library_paths()
    set_seed(TRAIN_CONFIG["seed"])

    # 2. Training Phase
    # Initialize trainer (which loads data and model) and run the training loop
    trainer = Trainer()
    trainer.fit()

    # 3. Validation & Failure Analysis Phase
    print("\n=== Final Validation & Failure Analysis ===")

    # Load the best model saved during training
    if os.path.exists(PATHS["model_save_path"]):
        trainer.model.load_state_dict(
            torch.load(PATHS["model_save_path"], map_location=trainer.device)
        )
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    trainer.model.eval()

    # Collect predictions on the full validation set
    all_preds = []
    all_targets = []

    # Use the validation loader from the trainer
    # We iterate manually to collect raw predictions for analysis
    with torch.no_grad():
        for inputs, targets in trainer.val_loader:
            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device)

            outputs = trainer.model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metric
    final_acc = accuracy_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis: Correlation between Error and Duration
    # Construct a dataframe for analysis
    df_analysis = trainer.df_val.copy()

    # Ensure alignment (DataLoader preserves order if shuffle=False, which it is for val)
    if len(df_analysis) != len(all_preds):
        print(
            "Warning: Mismatch in validation set size and prediction count. Skipping detailed analysis."
        )
    else:
        df_analysis["pred"] = all_preds
        df_analysis["target"] = all_targets
        df_analysis["error"] = (df_analysis["pred"] != df_analysis["target"]).astype(
            int
        )

        # Calculate durations
        # Note: We use library.config.INPUT_DIR because we patched the paths logic
        full_paths = [
            os.path.join(library.config.INPUT_DIR, p) for p in df_analysis["filepath"]
        ]
        df_analysis["duration"] = [get_audio_duration(p) for p in full_paths]

        # Correlation
        corr = df_analysis["error"].corr(df_analysis["duration"])
        print(f"Correlation between Error Magnitude and Input Duration: {corr}")

    # 4. Submission Phase
    THRESHOLD = 0.9611927398444252

    if final_acc > THRESHOLD:
        print(
            f"Validation accuracy meets threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"Validation accuracy ({final_acc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
