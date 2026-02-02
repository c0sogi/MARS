import os
import torch
import numpy as np
import nltk
from scipy.stats import pearsonr
import warnings

# Import library modules
from library.config import Config, set_seed
from library.data_loader import get_dataloaders
from library.model import BiLSTMClassifier
from library.trainer import run_training
from library.inference import run_inference
from library.utils import decode_predictions

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def get_truth_sequences(labels, lengths):
    """
    Decodes ground truth sequences from label tensors.
    Replicates logic from Trainer class for independent analysis.
    """
    labels_np = labels.cpu().numpy()
    lengths_np = lengths.cpu().numpy()
    truth_seqs = []

    for i in range(len(labels_np)):
        l = lengths_np[i]
        seq = labels_np[i, :l]

        # Collapse repeats and remove background (0)
        if len(seq) == 0:
            truth_seqs.append([])
            continue

        collapsed = [seq[0]]
        for k in range(1, len(seq)):
            if seq[k] != seq[k - 1]:
                collapsed.append(seq[k])

        final_seq = [int(x) for x in collapsed if x != 0]
        truth_seqs.append(final_seq)

    return truth_seqs


def main():
    # 1. Setup and Configuration
    # Adjust Config for fast baseline execution
    Config.NUM_EPOCHS = 40
    set_seed(Config.SEED)

    print(
        f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Data Loading
    print("Loading datasets...")
    # debug=False loads the full dataset (subject to limit=0 in get_dataloaders)
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 3. Model Training
    print("Initializing and training model...")
    model = BiLSTMClassifier()

    # run_training returns the trainer instance, but we will load the best checkpoint manually later
    run_training(model, train_loader, val_loader)

    # 4. Validation and Failure Analysis
    print("Performing final validation and failure analysis...")

    # Load the best model saved during training
    best_model = BiLSTMClassifier().to(Config.DEVICE)
    if os.path.exists(Config.MODEL_CHECKPOINT):
        best_model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Checkpoint not found. Using current model weights.")
        best_model = model

    best_model.eval()

    all_preds = []
    all_truth = []

    # Metrics for failure analysis
    sample_errors = []
    sample_lengths = []
    sample_motion_means = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(Config.DEVICE)
            labels = batch["labels"].to(Config.DEVICE)
            lengths = batch["lengths"].to(Config.DEVICE)

            # Forward pass
            logits = best_model(features, lengths)

            # Decode Predictions
            batch_preds = decode_predictions(logits)

            # Decode Ground Truth
            batch_truth = get_truth_sequences(labels, lengths)

            all_preds.extend(batch_preds)
            all_truth.extend(batch_truth)

            # Collect data for failure analysis
            feats_np = features.cpu().numpy()
            lengths_np = lengths.cpu().numpy()

            for i in range(len(batch_preds)):
                # 1. Error Magnitude (Levenshtein Distance)
                dist = nltk.edit_distance(batch_preds[i], batch_truth[i])
                sample_errors.append(dist)

                # 2. Input Feature: Sequence Length
                l = lengths_np[i]
                sample_lengths.append(l)

                # 3. Input Feature: Motion Magnitude
                # Velocity features are at indices 36 to 72 (36 dimensions)
                # We take the mean absolute velocity over the valid frames
                if l > 0:
                    valid_feats = feats_np[i, :l, 36:72]
                    motion_mag = np.mean(np.abs(valid_feats))
                else:
                    motion_mag = 0.0
                sample_motion_means.append(motion_mag)

    # Compute Final Validation Metric
    total_distance = sum(sample_errors)
    total_truth_length = sum([len(t) for t in all_truth])

    # Avoid division by zero
    final_metric = (
        total_distance / total_truth_length if total_truth_length > 0 else float("inf")
    )

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    if len(sample_errors) > 1:
        corr_len, _ = pearsonr(sample_errors, sample_lengths)
        corr_motion, _ = pearsonr(sample_errors, sample_motion_means)

        print("-" * 30)
        print("Failure Analysis (Correlation with Error Magnitude):")
        print(f"Sequence Length: {corr_len:.4f}")
        print(f"Motion Magnitude: {corr_motion:.4f}")
        print("-" * 30)

    # 5. Inference and Submission
    if final_metric < 0.4817:
        print("Generating submission for test set...")
        # run_inference handles loading the checkpoint, predicting, and saving to CSV
        run_inference(
            checkpoint_path=Config.MODEL_CHECKPOINT, output_path=Config.SUBMISSION_FILE
        )
    else:
        print(
            f"Validation metric {final_metric:.4f} did not meet threshold 0.4817. Skipping submission."
        )


if __name__ == "__main__":
    main()
