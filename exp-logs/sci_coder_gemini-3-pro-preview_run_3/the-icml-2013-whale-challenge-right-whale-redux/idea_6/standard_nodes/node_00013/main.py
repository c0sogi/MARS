import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_roc_auc
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.model import WhaleEfficientNetV2


def analyze_failure(val_df, preds, targets):
    """
    Performs failure analysis on the validation set by calculating the correlation
    between the model's error magnitude and input signal features (Duration, Amplitude).
    """
    print("Performing failure analysis...")

    # Calculate absolute error magnitude
    errors = np.abs(targets - preds)

    durations = []
    mean_amps = []

    # Iterate through validation files to extract basic signal features
    # val_loader is not shuffled, so it aligns with val_df rows
    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Read audio file
            data, sr = sf.read(full_path)

            # Feature 1: Duration
            durations.append(len(data) / sr)

            # Feature 2: Mean Amplitude (proxy for loudness/signal-to-noise ratio)
            if data.ndim > 1:
                data = data.mean(axis=1)
            mean_amps.append(np.mean(np.abs(data)))

        except Exception:
            # Fallback for any unreadable files
            durations.append(0.0)
            mean_amps.append(0.0)

    durations = np.array(durations)
    mean_amps = np.array(mean_amps)

    # Calculate Pearson Correlation Coefficient
    # Check for non-zero variance to avoid division by zero
    if np.std(durations) > 1e-6:
        corr_dur = np.corrcoef(errors, durations)[0, 1]
    else:
        corr_dur = 0.0

    if np.std(mean_amps) > 1e-6:
        corr_amp = np.corrcoef(errors, mean_amps)[0, 1]
    else:
        corr_amp = 0.0

    print(f"Error Correlation with Duration: {corr_dur:.4f}")
    print(f"Error Correlation with Mean Amplitude: {corr_amp:.4f}")


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for hardware optimization and task constraints
    Config.BATCH_SIZE = 64  # Increase batch size for A100 GPU
    Config.EPOCHS = 5  # Limit epochs for a fast baseline

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Initializing DataLoaders...")
    # debug=False ensures we use the full dataset to achieve the high metric threshold
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # =========================================================================
    # 3. Model Training
    # =========================================================================
    print("Starting training pipeline...")
    trainer = Trainer(train_loader, val_loader)
    trainer.fit()

    # =========================================================================
    # 4. Model Reloading
    # =========================================================================
    # Explicitly reload the best checkpoint saved during training
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Reloading best model weights from {best_model_path}...")
        state_dict = torch.load(best_model_path, map_location=device)
        trainer.model.load_state_dict(state_dict)
    else:
        print("Warning: Best model checkpoint not found. Using final model state.")

    trainer.model.eval()

    # =========================================================================
    # 5. Validation Inference & Metrics
    # =========================================================================
    print("Running inference on validation set...")
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Disable gradients and use mixed precision for fast inference
            with torch.amp.autocast("cuda"):
                logits = trainer.model(images)
                probs = torch.sigmoid(logits)

            val_preds.append(probs.float().cpu().numpy())
            val_targets.append(labels.float().numpy())

    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()

    final_auc = compute_roc_auc(val_targets, val_preds)

    # Required Output
    print(f"Final Validation Metric: {final_auc}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    val_df = pd.read_csv(Config.VAL_CSV)
    analyze_failure(val_df, val_preds, val_targets)

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    TARGET_THRESHOLD = 0.9947886445829673

    if final_auc > TARGET_THRESHOLD:
        print(f"Metric {final_auc} > {TARGET_THRESHOLD}. Generating submission...")

        test_preds = []
        test_clips = []

        with torch.no_grad():
            for images, clip_names in test_loader:
                images = images.to(device)
                with torch.amp.autocast("cuda"):
                    logits = trainer.model(images)
                    probs = torch.sigmoid(logits)

                test_preds.append(probs.float().cpu().numpy())
                test_clips.extend(clip_names)

        test_preds = np.concatenate(test_preds).flatten()

        # Create submission DataFrame
        submission = pd.DataFrame({"clip": test_clips, "probability": test_preds})

        # Save to file
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"Metric {final_auc} did not meet threshold {TARGET_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
