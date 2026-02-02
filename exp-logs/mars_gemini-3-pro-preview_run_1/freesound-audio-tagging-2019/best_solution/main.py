import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.trainer import Trainer
from library.utils import set_seed, load_checkpoint, calculate_lwlrap
from library.dataset import AudioDataset
from torch.utils.data import DataLoader


def calculate_sample_lwlrap(truth, scores):
    """
    Calculates LWLRAP for each sample individually.
    Returns a numpy array of scores.
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape

    # Sort scores descending
    sorted_indices = np.argsort(-scores, axis=1)
    sorted_truth = np.take_along_axis(truth, sorted_indices, axis=1)

    # Cumulative true positives
    cumulative_tp = np.cumsum(sorted_truth, axis=1)
    ranks = np.arange(1, num_classes + 1)
    precisions = cumulative_tp / ranks

    relevant_precisions = precisions * sorted_truth

    # Sum of precisions for relevant labels per sample
    sum_precisions = relevant_precisions.sum(axis=1)

    # Count of relevant labels per sample
    label_counts = truth.sum(axis=1)

    # Avoid division by zero
    safe_counts = np.maximum(label_counts, 1)

    return sum_precisions / safe_counts


def run_inference(model, loader, device):
    """
    Runs inference and returns raw predictions and targets.
    """
    model.eval()
    all_preds = []
    all_targets = []  # Only for val

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                all_targets.append(labels.cpu())
            else:
                # Test set might only return images if dataset was designed that way,
                # but AudioDataset returns (spec, target) even for test (target is placeholder)
                images, labels = batch
                images = images.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    if all_targets:
        all_targets = torch.cat(all_targets, dim=0).numpy()
        return all_preds, all_targets
    return all_preds, None


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Training
    print("\n=== Starting Training ===")
    trainer = Trainer()
    trainer.train()

    # 3. Load Best Model
    print("\n=== Loading Best Model ===")
    best_score = load_checkpoint(trainer.model, filename="best_model.pth")
    print(f"Checkpoint loaded. Best recorded score: {best_score}")

    # 4. Validation
    print("\n=== Running Validation ===")
    val_loader = trainer.get_dataloader("val")

    # Run inference
    val_preds, val_targets = run_inference(trainer.model, val_loader, device)

    # Calculate Final Metric
    final_metric = calculate_lwlrap(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate per-sample error (1 - LWLRAP)
    sample_scores = calculate_sample_lwlrap(val_targets, val_preds)
    sample_errors = 1.0 - sample_scores

    # Get metadata for correlation features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Feature 1: Label Count (Cardinality)
    label_counts = val_targets.sum(axis=1)

    # Feature 2: Audio Duration
    # We need to read this from files as it's not in the CSV explicitly
    durations = []
    for filepath in val_df["filepath"]:
        full_path = os.path.join(Config.INPUT_ROOT, filepath)
        try:
            info = sf.info(full_path)
            durations.append(info.duration)
        except:
            durations.append(0.0)
    durations = np.array(durations)

    # Correlations
    # Filter out any potential NaNs or zeros if file read failed
    valid_mask = durations > 0

    if np.sum(valid_mask) > 0:
        corr_duration, _ = pearsonr(sample_errors[valid_mask], durations[valid_mask])
        corr_labels, _ = pearsonr(sample_errors, label_counts)

        print(f"Correlation (Error vs Duration): {corr_duration:.4f}")
        print(f"Correlation (Error vs Label Count): {corr_labels:.4f}")
    else:
        print("Could not compute correlations due to missing audio data.")

    # 6. Submission
    THRESHOLD = 0.8555

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        # Using batch_size=1 for safety with variable length audio
        test_dataset = AudioDataset(split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        test_preds, _ = run_inference(trainer.model, test_loader, device)

        # Create Submission DataFrame
        # Load sample submission to get column names and order
        sample_sub = pd.read_csv(
            os.path.join(Config.INPUT_ROOT, "sample_submission.csv")
        )

        # The AudioDataset loads classes based on train.csv columns.
        # We verified in the prompt analysis that train.csv columns match sample_submission columns [1:]
        # So we can directly assign predictions.

        submission = pd.DataFrame(test_preds, columns=sample_sub.columns[1:])
        submission.insert(0, "fname", sample_sub["fname"])

        # Save
        save_path = Config.SUBMISSION_PATH
        submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
