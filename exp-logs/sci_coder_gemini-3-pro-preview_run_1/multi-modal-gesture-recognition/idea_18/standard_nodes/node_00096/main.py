import os
import sys
import numpy as np
import torch
import pandas as pd
import scipy.stats
import warnings
import nltk

# Suppress warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# Monkey-patch tqdm to suppress progress bars
# ---------------------------------------------------------
def silent_tqdm(iterable, *args, **kwargs):
    return iterable


import tqdm

tqdm.tqdm = silent_tqdm

# Import library modules
# We patch the tqdm reference in library.trainer to ensure it uses the silent version
import library.trainer

library.trainer.tqdm = silent_tqdm

from library.config import Config
from library.utils import set_seed, apply_median_filter, decode_predictions
from library.trainer import Trainer


def main():
    # 1. Setup
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Optimize workers for the environment
    Config.NUM_WORKERS = 12

    print("Initializing Trainer...")
    trainer = Trainer()

    # 2. Training
    print("Starting Training...")
    trainer.run()

    # 3. Validation & Failure Analysis
    print("Running Validation and Failure Analysis...")

    # Load best model
    if os.path.exists(trainer.best_model_path):
        trainer.model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=trainer.device)
        )
    else:
        print("Warning: No best model found, using current state.")

    trainer.model.eval()
    val_loader = trainer.val_loader

    total_distance = 0
    total_gestures = 0

    errors = []
    durations = []
    seq_lengths = []

    with torch.no_grad():
        for skeleton, audio, labels, lengths in val_loader:
            skeleton = skeleton.to(trainer.device)
            audio = audio.to(trainer.device)
            labels = labels.to(trainer.device)
            lengths = lengths.to(trainer.device)

            # Inference
            outputs = trainer.model(skeleton, audio, lengths)
            preds = torch.argmax(outputs, dim=2).cpu().numpy()
            targets = labels.cpu().numpy()

            # Iterate batch
            for i in range(len(preds)):
                length = lengths[i].item()
                pred_raw = preds[i, :length]
                target_raw = targets[i, :length]

                # Post-processing
                pred_smooth = apply_median_filter(
                    pred_raw, window_size=Config.MEDIAN_FILTER_WINDOW
                )
                pred_seq = decode_predictions(
                    pred_smooth,
                    min_segment_length=Config.MIN_SEGMENT_LENGTH,
                    background_class_id=Config.BACKGROUND_CLASS_ID,
                )

                # Ground Truth Decoding (min_segment=1 to capture all intended gestures)
                target_seq = decode_predictions(
                    target_raw,
                    min_segment_length=1,
                    background_class_id=Config.BACKGROUND_CLASS_ID,
                )

                # Metric Calculation
                dist = nltk.edit_distance(pred_seq, target_seq)
                n_gestures = len(target_seq)

                total_distance += dist
                total_gestures += n_gestures

                # Failure Analysis Data
                errors.append(dist)
                durations.append(length)
                seq_lengths.append(n_gestures)

    # Compute Final Metric
    final_metric = total_distance / total_gestures if total_gestures > 0 else 0.0

    # Print Metric (Full Precision)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis Correlations
    if len(errors) > 1:
        # Avoid constant input warning if variance is 0
        if np.std(errors) > 0 and np.std(durations) > 0:
            corr_dur, _ = scipy.stats.pearsonr(errors, durations)
        else:
            corr_dur = 0.0

        if np.std(errors) > 0 and np.std(seq_lengths) > 0:
            corr_seq, _ = scipy.stats.pearsonr(errors, seq_lengths)
        else:
            corr_seq = 0.0

        print("Failure Analysis - Correlations with Error:")
        print(f"  Duration (Frames): {corr_dur}")
        print(f"  Sequence Length (Gestures): {corr_seq}")

    # 4. Submission
    THRESHOLD = 0.0765306122
    if final_metric < THRESHOLD:
        print(f"Metric is below threshold ({THRESHOLD}). Generating submission...")
        trainer.generate_submission()
    else:
        print(
            f"Metric ({final_metric}) is above threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
