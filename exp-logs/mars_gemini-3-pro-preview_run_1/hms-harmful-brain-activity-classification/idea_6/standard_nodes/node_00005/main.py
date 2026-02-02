import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, kl_divergence_score
from library.data import get_dataloaders
from library.model import SymmetryAwareNet
from library.train import Trainer


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    Config.init_directories()

    # Override Config for Fast Baseline
    # A100 can handle larger batches, speeding up training
    Config.BATCH_SIZE = 64
    Config.EPOCHS = 3

    # Limit training data size for speed (Fast Baseline requirement)
    # We keep validation full to ensure accurate metric calculation
    TRAIN_DEBUG_SIZE = 20000

    print(
        f"Configuration: Batch Size={Config.BATCH_SIZE}, Epochs={Config.EPOCHS}, Device={Config.DEVICE}"
    )

    # 2. Data Loading
    print("Loading DataLoaders...")
    # We pass debug_size to limit training data, but we need full validation data for the metric
    # get_dataloaders applies debug_size to all if provided, so we handle it manually or
    # just accept that we need to construct loaders carefully.
    # The library function applies debug_size to all. To strictly follow the plan:
    # We will load data normally, then subset the training loader's dataset indices if needed,
    # or just rely on the library function if we accept a smaller val set for the "baseline run".
    # However, the prompt requires "Final Validation Metric computed on the entire hold-out validation set".
    # So we cannot use debug_size in get_dataloaders for validation.

    # Workaround: Load full datasets, then manually subset train_loader or just use a subset sampler.
    # Since we can't easily modify the library function's internal logic without copying,
    # we will load full data and use itertools.islice or similar in the training loop?
    # No, cleaner to just use the full training set but fewer epochs (3 epochs on 80k is fast on A100).
    # 80k / 64 batch size = 1250 steps. 3 epochs = 3750 steps. This is very fast (mins).
    # So we will NOT use debug_size to ensure full validation integrity, relying on the A100 speed.

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug_size=None,  # Use full data, rely on low epochs and powerful GPU
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = SymmetryAwareNet()

    # 4. Training
    print("Starting Training...")
    trainer = Trainer(model, Config.DEVICE)

    # We use the fit method from the library
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 5. Validation & Metric Calculation
    print("Performing Final Validation...")

    # Load best model
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    # Inference on Validation Set
    # We use trainer.predict which returns (probs, ids)
    # Note: trainer.predict expects 'eeg_id' in batch, which our dataset provides.
    val_probs, val_ids = trainer.predict(val_loader)

    # Get Ground Truth
    # Since shuffle=False for val_loader, the order matches the metadata CSV
    val_metadata = pd.read_csv(Config.VAL_CSV)
    val_targets = val_metadata[Config.TARGET_COLS].values

    # Compute Metric
    final_metric = kl_divergence_score(val_targets, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")

    # Calculate KL per sample
    # Clip predictions
    epsilon = 1e-15
    y_pred = np.clip(val_probs, epsilon, 1 - epsilon)
    y_true = val_targets

    # KL = sum(P * log(P/Q))
    # Handle P=0 case safely
    term_true = np.where(y_true > 0, y_true * np.log(y_true), 0.0)
    term_pred = y_true * np.log(y_pred)
    kl_per_sample = np.sum(term_true - term_pred, axis=1)

    # Correlate with metadata
    # Ensure lengths match
    if len(kl_per_sample) == len(val_metadata):
        # EEG Offset
        if "eeg_label_offset_seconds" in val_metadata.columns:
            corr_eeg, _ = pearsonr(
                val_metadata["eeg_label_offset_seconds"].fillna(0), kl_per_sample
            )
            print(f"Correlation between Error (KL) and EEG Offset: {corr_eeg:.4f}")

        # Spectrogram Offset
        if "spectrogram_label_offset_seconds" in val_metadata.columns:
            corr_spec, _ = pearsonr(
                val_metadata["spectrogram_label_offset_seconds"].fillna(0),
                kl_per_sample,
            )
            print(
                f"Correlation between Error (KL) and Spectrogram Offset: {corr_spec:.4f}"
            )
    else:
        print(
            "Warning: Mismatch in validation samples and metadata length. Skipping detailed failure analysis."
        )

    # 7. Submission Logic
    THRESHOLD = 0.7327804565429688

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        test_probs, test_ids = trainer.predict(test_loader)

        # Create DataFrame
        # Map config target cols to submission format
        # Config.TARGET_COLS are ['seizure_prob', ...]
        # Submission requires ['seizure_vote', ...]
        submission_cols = [
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]

        submission_df = pd.DataFrame(test_probs, columns=submission_cols)
        submission_df.insert(0, "eeg_id", test_ids)

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
