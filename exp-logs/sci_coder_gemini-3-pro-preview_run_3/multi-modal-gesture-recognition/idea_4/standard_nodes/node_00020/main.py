import sys
import os
import torch
import numpy as np

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.inference import generate_test_predictions
from library.utils import decode_predictions


def levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences of integers.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override epochs for a fast baseline execution
    Config.NUM_EPOCHS = 15

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets...")
    # Use cached data to speed up loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # ==========================================
    # 3. Training
    # ==========================================
    print("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader, device=device)

    print("Starting training...")
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # ==========================================
    # 4. Validation & Metric Calculation
    # ==========================================
    print("Performing validation and metric calculation...")
    trainer.model.eval()

    total_lev_distance = 0.0
    total_gt_gestures = 0

    # Storage for failure analysis
    sample_errors = []
    sample_lengths = []
    sample_num_gestures = []

    with torch.no_grad():
        for batch_idx, (features, labels, sample_id) in enumerate(val_loader):
            features = features.to(device)
            # labels is (Batch=1, Time)

            # Forward pass
            _, s2_logits = trainer.model(features)

            # Decode Predictions
            # s2_logits: (1, Time, Classes) -> (Time, Classes)
            logits_seq = s2_logits.squeeze(0)
            pred_seq = decode_predictions(logits_seq, min_len=5)

            # Decode Ground Truth
            # labels: (1, Time) -> (Time,)
            gt_seq_dense = labels.squeeze(0).cpu().numpy()
            # Use min_len=1 for GT to ensure we capture all annotated gestures
            gt_seq = decode_predictions(gt_seq_dense, min_len=1)

            # Calculate Distance
            dist = levenshtein_distance(pred_seq, gt_seq)

            # Accumulate
            total_lev_distance += dist
            total_gt_gestures += len(gt_seq)

            # Store stats
            sample_errors.append(dist)
            sample_lengths.append(features.shape[1])
            sample_num_gestures.append(len(gt_seq))

    # Calculate final metric
    # Metric = Total Levenshtein Distance / Total Number of GT Gestures
    if total_gt_gestures > 0:
        final_metric = total_lev_distance / total_gt_gestures
    else:
        final_metric = float("inf")

    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    if len(sample_errors) > 1:
        # Calculate correlations using numpy
        # corrcoef returns matrix [[1, r], [r, 1]]
        corr_len = np.corrcoef(sample_errors, sample_lengths)[0, 1]
        corr_num = np.corrcoef(sample_errors, sample_num_gestures)[0, 1]

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        # Interpretation
        if abs(corr_len) > 0.3:
            print(">> Significant correlation with sequence length detected.")
        if abs(corr_num) > 0.3:
            print(">> Significant correlation with gesture density detected.")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # ==========================================
    # 6. Submission
    # ==========================================
    threshold = 0.30627871362940273

    if final_metric < threshold:
        print(
            f"\nValidation metric ({final_metric:.4f}) is better than threshold ({threshold:.4f})."
        )
        print("Generating submission file...")
        generate_test_predictions(load_cached_data=True, device=device)
    else:
        print(
            f"\nValidation metric ({final_metric:.4f}) did not meet threshold ({threshold:.4f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
