import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import VAL_METADATA_PATH, SEED, MIN_DURATION, WORKING_DIR
from library.model import ASH_KN, predict_sequence
from library.trainer import run_training_session
from library.data_loader import load_and_process_data
from library.utils import run_length_encoding, levenshtein_distance
from library.inference import generate_submission


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Training
    # We limit epochs to 25 to ensure full convergence (Cite {solution_lesson_node_00091})
    print("Starting Training Session...")
    best_model_path = run_training_session(
        epochs=25, batch_size=32, load_cached_data=True
    )
    print(f"Best model saved at: {best_model_path}")

    # 3. Validation & Failure Analysis
    print("Starting Validation and Failure Analysis...")

    # Load Validation Data
    val_data = load_and_process_data(
        VAL_METADATA_PATH, "dataset_val", load_cached_data=True
    )

    # Load Best Model
    model = ASH_KN().to(device)
    if not os.path.exists(best_model_path):
        print("Error: Best model file not found.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Storage for analysis
    sample_errors = []
    sample_durations = []
    sample_num_gestures = []

    total_distance = 0
    total_ref_length = 0

    # Inference Loop
    with torch.no_grad():
        # Sort keys for deterministic order
        for sid in sorted(val_data.keys()):
            sample = val_data[sid]
            skeleton = sample["skeleton"]
            audio = sample["audio"]
            labels_frame_wise = sample["labels"]

            # Predict
            frame_preds = predict_sequence(model, skeleton, audio, device)

            # Decode Hypothesis
            hyp_seq = run_length_encoding(frame_preds, min_duration=MIN_DURATION)

            # Decode Reference (Ground Truth)
            # Use min_duration=1 for GT to capture all annotated segments
            ref_seq = run_length_encoding(labels_frame_wise, min_duration=1)

            # Compute Metric for this sample
            dist = levenshtein_distance(hyp_seq, ref_seq)

            # Accumulate for Global Score
            total_distance += dist
            total_ref_length += len(ref_seq)

            # Collect data for Failure Analysis
            sample_errors.append(dist)
            sample_durations.append(skeleton.shape[0])  # Duration in frames
            sample_num_gestures.append(len(ref_seq))

    # Compute Final Metric
    if total_ref_length > 0:
        final_metric = total_distance / total_ref_length
    else:
        final_metric = 1.0  # Fallback if no gestures in validation set

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("-" * 30)
    print("Failure Analysis:")

    if len(sample_errors) > 1:
        # Correlation: Error vs Duration
        corr_dur, _ = pearsonr(sample_errors, sample_durations)
        print(f"Correlation (Error vs Sequence Duration): {corr_dur:.4f}")

        # Correlation: Error vs Num Gestures
        corr_num, _ = pearsonr(sample_errors, sample_num_gestures)
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        # Additional Insight: Error per Gesture count
        # Normalize error by number of gestures to see if longer sequences are harder proportionally
        normalized_errors = np.array(sample_errors) / np.maximum(
            np.array(sample_num_gestures), 1
        )
        corr_norm_dur, _ = pearsonr(normalized_errors, sample_durations)
        print(f"Correlation (Normalized Error vs Duration): {corr_norm_dur:.4f}")
    else:
        print("Not enough samples for correlation analysis.")
    print("-" * 30)

    # 5. Submission
    THRESHOLD = 0.2251
    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(best_model_path, submission_filename="submission.csv")
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
