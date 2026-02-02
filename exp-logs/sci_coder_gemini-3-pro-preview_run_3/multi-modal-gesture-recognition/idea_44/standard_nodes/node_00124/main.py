import os
import sys
import torch
import numpy as np
import warnings
from scipy.stats import pearsonr

# Import provided library components
from library.config import Config
from library.train import train_model
from library.predict import generate_submission
from library.dataset import GestureDataset
from library.model import RCMCN
from library.utils import (
    decode_predictions_to_labels,
    compute_levenshtein,
    run_length_encoding,
)


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 1. Setup & Configuration Overrides
    Config.set_seed(Config.SEED)
    Config.setup_dirs()

    # Override hyperparameters for Fast Baseline execution
    # Reducing epochs to ensure completion within time limits while sufficient for the small dataset
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 32

    # 2. Train the Model
    # train_model returns the best validation score (Mean Levenshtein Distance)
    best_score = train_model(load_cached_data=True)

    # 3. Report Metric
    print(f"Final Validation Metric: {best_score}")

    # 4. Failure Analysis
    print("Performing Failure Analysis on Validation Set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Validation Data (Full sequences, batch_size=1)
    val_dataset = GestureDataset(split="val", load_cached_data=True, transform=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    # Load Best Model
    model = RCMCN().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print(
            "Warning: Best model checkpoint not found. Skipping detailed failure analysis."
        )
        return

    model.eval()

    errors = []
    lengths = []
    num_gestures = []

    with torch.no_grad():
        for i, (features, labels) in enumerate(val_loader):
            features = features.to(device)
            # labels shape: (1, T)

            # Forward pass
            _, _, logits3 = model(features)
            probs = torch.softmax(logits3, dim=2)
            frame_probs = probs.squeeze(0).cpu().numpy()

            # Decode Predictions
            pred_labels = decode_predictions_to_labels(frame_probs)

            # Decode Ground Truth
            gt_frame_labels = labels.squeeze(0).cpu().numpy()
            gt_segments = run_length_encoding(gt_frame_labels)
            gt_labels = [seg["label"] for seg in gt_segments if seg["label"] != 0]

            # Compute Metric for this sample
            dist = compute_levenshtein(gt_labels, pred_labels)

            # Calculate Error Rate (Distance / Num Gestures)
            # If Num Gestures is 0, error is just the distance (insertions)
            gt_len = len(gt_labels)
            if gt_len > 0:
                error_rate = dist / gt_len
            else:
                error_rate = float(dist)

            errors.append(error_rate)
            lengths.append(features.shape[1])
            num_gestures.append(gt_len)

    # Calculate Correlations
    if len(errors) > 1:
        errors_np = np.array(errors)
        lengths_np = np.array(lengths)
        gestures_np = np.array(num_gestures)

        # Check for non-zero variance to avoid warnings
        if np.std(errors_np) > 1e-9 and np.std(lengths_np) > 1e-9:
            corr_len, _ = pearsonr(errors_np, lengths_np)
            print(f"Correlation (Error vs Sequence Length): {corr_len}")
        else:
            print("Correlation (Error vs Sequence Length): Undefined (constant values)")

        if np.std(errors_np) > 1e-9 and np.std(gestures_np) > 1e-9:
            corr_num, _ = pearsonr(errors_np, gestures_np)
            print(f"Correlation (Error vs Num Gestures): {corr_num}")
        else:
            print("Correlation (Error vs Num Gestures): Undefined (constant values)")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 5. Submission Generation
    threshold = 0.1860643185298622
    if best_score < threshold:
        print(
            f"Validation score {best_score} meets threshold {threshold}. Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation score {best_score} does not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
