import sys
import os
import torch
import numpy as np
import scipy.stats
import nltk
import warnings

# Ensure the current directory is in the python path to import the library correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.trainer import run_training
from library.inference import generate_submission, decode_predictions
from library.data_loader import get_dataloaders
from library.model import CRGCN

# Suppress warnings
warnings.filterwarnings("ignore")


def decode_targets(targets, lengths):
    """
    Decodes frame-wise targets into a list of gesture sequences.
    Collapses repeats and removes background class (0).
    """
    targets_np = targets.cpu().numpy()
    decoded_sequences = []

    for i in range(targets_np.shape[0]):
        length = lengths[i]
        raw_target = targets_np[i, :length]

        sequence = []
        prev_label = -1

        for label in raw_target:
            if label != prev_label:
                if label != 0:
                    sequence.append(int(label))
                prev_label = label

        decoded_sequences.append(sequence)

    return decoded_sequences


def main():
    # 1. Setup
    set_seed(Config.SEED)
    logger = setup_logger("runfile")

    # 2. Train
    # We use 30 epochs to ensure a good balance between speed and convergence
    # for this fast baseline.
    logger.info("Starting training pipeline...")
    run_training(epochs=30)

    # 3. Validation & Failure Analysis
    logger.info("Starting validation and failure analysis...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Validation Data
    # get_dataloaders returns (train, val, test)
    _, val_loader, _ = get_dataloaders()

    # Load Best Model
    model = CRGCN().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}. Aborting.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_predictions = []
    all_targets = []
    sample_errors = []
    sample_lengths = []

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            features, labels, boundaries, mask, lengths = batch
            features = features.to(device)
            mask = mask.to(device)

            # Forward pass
            outputs = model(features, mask)
            # Extract Stage 3 logits (Final Refinement)
            stage3_logits = outputs["stage3"][0]

            # Decode
            batch_preds = decode_predictions(stage3_logits, lengths)
            batch_targets = decode_targets(labels, lengths)

            all_predictions.extend(batch_preds)
            all_targets.extend(batch_targets)

            # Collect metrics for failure analysis
            for pred, target, length in zip(batch_preds, batch_targets, lengths):
                # Compute Levenshtein distance for this sample
                dist = nltk.edit_distance(pred, target)
                sample_errors.append(dist)
                sample_lengths.append(length.item())

    # Compute Final Metric
    # Metric = Sum of Levenshtein distances / Total number of gestures in truth
    total_distance = sum(sample_errors)
    total_gestures = sum(len(t) for t in all_targets)

    final_metric = total_distance / total_gestures if total_gestures > 0 else 0.0

    # Print Final Metric (Required Format)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Correlation between error magnitude and sequence length
    if len(sample_errors) > 1:
        corr, _ = scipy.stats.pearsonr(sample_errors, sample_lengths)
        print(
            f"Failure Analysis: Correlation between error magnitude and sequence length: {corr:.4f}"
        )

    # 4. Submission
    # Threshold check
    threshold = 0.06789606035205364

    if final_metric < threshold:
        logger.info(
            f"Validation metric {final_metric} is lower than threshold {threshold}. Generating submission..."
        )
        generate_submission()
    else:
        logger.info(
            f"Validation metric {final_metric} is not lower than threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
