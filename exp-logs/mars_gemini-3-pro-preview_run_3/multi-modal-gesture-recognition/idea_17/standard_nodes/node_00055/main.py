import os
import torch
import numpy as np
import pandas as pd
import scipy.stats
from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.data_loader import load_data, KinematicAugmentor
from library.model import KC_IRN
from library.train import train_model, decode_predictions
from library.inference import generate_submission


def evaluate_and_analyze(model, val_data, device):
    """
    Evaluates the model on the validation set, computes the final metric,
    and performs failure analysis (correlation of error with input length).
    """
    model.eval()

    all_preds = []
    all_targets = []

    # Lists for failure analysis
    sample_errors = []
    sample_lengths = []  # Number of frames

    window_size = Config.WINDOW_SIZE
    stride = Config.TEST_STRIDE

    print("Starting detailed validation and failure analysis...")

    with torch.no_grad():
        for sample in val_data:
            # --- Feature Preparation (Same as inference.py/train.py) ---
            raw_skel = sample["skeleton"]
            kinematics = KinematicAugmentor.compute_kinematics(raw_skel)
            audio = sample["audio"]
            full_features = np.concatenate([kinematics, audio], axis=1)
            T = full_features.shape[0]

            # Store length for analysis
            sample_lengths.append(T)

            # --- Inference (Sliding Window) ---
            prob_buffer = np.zeros((T, Config.NUM_CLASSES), dtype=np.float32)
            count_buffer = np.zeros((T, 1), dtype=np.float32)

            # Handle short sequences
            if T < window_size:
                pad_len = window_size - T
                feat_padded = np.pad(full_features, ((0, pad_len), (0, 0)), mode="edge")
                feat_tensor = (
                    torch.from_numpy(feat_padded).float().unsqueeze(0).to(device)
                )

                outputs = model(feat_tensor)
                final_logits = outputs[-1]
                probs = torch.softmax(final_logits, dim=2).cpu().numpy().squeeze(0)

                prob_buffer += probs[:T]
                count_buffer += 1.0
            else:
                # Sliding window
                for start in range(0, T - window_size + 1, stride):
                    end = start + window_size
                    window_feat = full_features[start:end]
                    feat_tensor = (
                        torch.from_numpy(window_feat).float().unsqueeze(0).to(device)
                    )

                    outputs = model(feat_tensor)
                    final_logits = outputs[-1]
                    probs = torch.softmax(final_logits, dim=2).cpu().numpy().squeeze(0)

                    prob_buffer[start:end] += probs
                    count_buffer[start:end] += 1.0

                # Last window
                last_start = T - window_size
                if last_start > 0 and (last_start % stride != 0):
                    window_feat = full_features[last_start:T]
                    feat_tensor = (
                        torch.from_numpy(window_feat).float().unsqueeze(0).to(device)
                    )

                    outputs = model(feat_tensor)
                    final_logits = outputs[-1]
                    probs = torch.softmax(final_logits, dim=2).cpu().numpy().squeeze(0)

                    prob_buffer[last_start:T] += probs
                    count_buffer[last_start:T] += 1.0

            # Average and Decode
            count_buffer[count_buffer == 0] = 1.0
            avg_probs = prob_buffer / count_buffer
            pred_dense = np.argmax(avg_probs, axis=1)
            pred_seq = decode_predictions(pred_dense)

            # Ground Truth
            gt_dense = sample["labels"]
            gt_seq = decode_predictions(gt_dense)

            all_preds.append(pred_seq)
            all_targets.append(gt_seq)

            # Compute Levenshtein for this sample
            # We need to implement the distance calculation for a single pair to store it
            # Using the utility function logic:
            from library.utils import levenshtein_distance

            dist = levenshtein_distance(pred_seq, gt_seq)
            sample_errors.append(dist)

    # --- Compute Final Metric ---
    # Metric = Sum(Errors) / Sum(GT Gestures)
    total_error = sum(sample_errors)
    total_gestures = sum(len(t) for t in all_targets)

    final_metric = total_error / total_gestures if total_gestures > 0 else 0.0

    # Print exactly as required
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Correlation between Error Magnitude and Sequence Length
    if len(sample_errors) > 1:
        corr, p_val = scipy.stats.pearsonr(sample_lengths, sample_errors)
        print(
            f"Failure Analysis - Correlation (Length vs Error): {corr:.4f} (p={p_val:.4f})"
        )
    else:
        print("Insufficient samples for correlation analysis.")

    return final_metric


def main():
    # 1. Setup
    set_seed()
    device = torch.device(Config.DEVICE)

    # 2. Train the model
    # This uses the library function which handles loops, saving, etc.
    print("=== Phase 1: Training ===")
    train_model()

    # 3. Load the best model for analysis
    print("\n=== Phase 2: Evaluation & Analysis ===")
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Model file not found.")
        return

    model = KC_IRN().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Load validation data
    val_data = load_data("val", load_cached_data=True)

    # Run evaluation
    final_metric = evaluate_and_analyze(model, val_data, device)

    # 4. Submission
    print("\n=== Phase 3: Submission ===")
    threshold = 0.2251
    if final_metric < threshold:
        print(
            f"Metric {final_metric} meets threshold ({threshold}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Metric {final_metric} does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
