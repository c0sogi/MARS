import os
import torch
import numpy as np
import pandas as pd
from nltk import edit_distance

# Import configuration and monkey-patch for faster execution
import library.config

library.config.NUM_EPOCHS = 25  # Reduce epochs for fast baseline execution

from library.config import set_seed, WORKING_DIR
from library.data_loader import get_data_loaders
from library.model import BiGRUEncoder
from library.train import train_model
from library.utils import decode_predictions, compute_levenshtein_ratio, save_submission


def main():
    # 1. Setup
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Training
    # train_model handles the training loop, validation monitoring, and saving the best model.
    print("Starting training pipeline...")
    best_model_path = train_model()
    print(f"Best model saved at: {best_model_path}")

    # 3. Validation & Failure Analysis
    print("Starting validation and failure analysis...")

    # Load the best model
    model = BiGRUEncoder().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get data loaders (using cached data)
    _, val_loader, test_loader = get_data_loaders()

    val_preds = []
    val_truths = []
    val_errors = []
    val_lengths = []
    val_num_gestures = []

    with torch.no_grad():
        for inputs, targets, sample_id in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            logits = model(inputs)

            # Decode predictions
            # logits: (1, Time, NumClasses) -> squeeze to (Time, NumClasses)
            logits_seq = logits.squeeze(0)
            pred_seq = decode_predictions(logits_seq)

            # Decode ground truth
            # targets: (1, Time) -> squeeze to (Time,)
            target_seq_raw = targets.squeeze(0).cpu().numpy()
            truth_seq = decode_predictions(target_seq_raw)

            val_preds.append(pred_seq)
            val_truths.append(truth_seq)

            # Collect metrics for failure analysis
            # Calculate Levenshtein distance for this sample
            dist = edit_distance(pred_seq, truth_seq)
            val_errors.append(dist)

            # Input length (Time dimension)
            val_lengths.append(inputs.shape[1])

            # Number of ground truth gestures
            val_num_gestures.append(len(truth_seq))

    # Compute Final Validation Metric
    final_metric = compute_levenshtein_ratio(val_preds, val_truths)
    print(f"Final Validation Metric: {final_metric}")

    # Perform Failure Analysis
    # Calculate correlation between error magnitude and input features
    if len(val_errors) > 1:
        # Correlation with Sequence Length
        corr_len = np.corrcoef(val_errors, val_lengths)[0, 1]
        # Correlation with Number of Gestures
        corr_num = np.corrcoef(val_errors, val_num_gestures)[0, 1]

        print("Failure Analysis Results:")
        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 4. Submission
    THRESHOLD = 0.32006125574272587

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for inputs, _, sample_id_tuple in test_loader:
                inputs = inputs.to(device)

                # Forward pass
                logits = model(inputs)

                # Decode
                logits_seq = logits.squeeze(0)
                pred_seq = decode_predictions(logits_seq)

                test_preds.append(pred_seq)
                # sample_id_tuple is a tuple with one element per batch (batch size is 1)
                test_ids.append(sample_id_tuple[0])

        # Save submission
        submission_path = "./submission/submission.csv"
        save_submission(test_preds, test_ids, submission_path)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
