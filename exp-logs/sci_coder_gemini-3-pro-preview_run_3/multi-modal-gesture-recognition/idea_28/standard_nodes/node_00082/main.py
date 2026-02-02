import os
import torch
import numpy as np
import pandas as pd
import random
import nltk
from library.config import Config
from library.trainer import Trainer
from library.inference import generate_submission, predict_sequence, decode_predictions
from library.utils import load_dataset, levenshtein_score, compute_kinematics
from library.model import RGHC_MN


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Training
    # We use a moderate number of epochs to ensure a fast baseline execution
    print("Initializing training...")
    trainer = Trainer(load_cached_data=True)
    trainer.fit(epochs=Config.NUM_EPOCHS)

    # 3. Validation & Metric Calculation
    print("Loading best model for validation...")
    model = RGHC_MN().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using current weights.")

    model.eval()

    print("Loading validation data...")
    val_data = load_dataset(
        Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH, load_cached_data=True
    )
    val_skeletons = val_data["skeletons"]
    val_audio = val_data["audio"]
    val_gt = val_data["gt_sequences"]

    all_preds = []
    sample_errors = []

    # Features for failure analysis
    feat_duration = []
    feat_num_gestures = []
    feat_avg_velocity = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for i, (skel, aud, gt) in enumerate(zip(val_skeletons, val_audio, val_gt)):
            # Predict
            probs = predict_sequence(model, skel, aud, device)
            pred_seq = decode_predictions(probs)
            all_preds.append(pred_seq)

            # Calculate sample-level Levenshtein distance for failure analysis
            # We use nltk.edit_distance for robust per-sample calculation
            dist = nltk.edit_distance(pred_seq, gt)

            # Normalize error by length of GT (avoid div by zero)
            # If GT is empty, error is the number of insertions (dist)
            denom = len(gt) if len(gt) > 0 else 1.0
            norm_error = dist / denom
            sample_errors.append(norm_error)

            # Extract features for analysis
            # 1. Duration (frames)
            feat_duration.append(skel.shape[0])

            # 2. Number of gestures
            feat_num_gestures.append(len(gt))

            # 3. Average Velocity (Kinematic intensity)
            # Compute kinematics: (T, 20, 9) -> [Pos, Vel, Acc]
            # Velocity is indices 3, 4, 5
            kin = compute_kinematics(skel)
            vel = kin[:, :, 3:6]  # (T, 20, 3)
            # Magnitude of velocity per joint per frame
            vel_mag = np.linalg.norm(vel, axis=2)  # (T, 20)
            # Mean velocity across all joints and time
            avg_vel = np.mean(vel_mag)
            feat_avg_velocity.append(avg_vel)

    # Calculate Global Metric
    # The metric is defined as Total Distance / Total GT Length
    final_metric = levenshtein_score(all_preds, val_gt)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    df_analysis = pd.DataFrame(
        {
            "error": sample_errors,
            "duration": feat_duration,
            "num_gestures": feat_num_gestures,
            "avg_velocity": feat_avg_velocity,
        }
    )

    # Compute correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error and Input Features:")
    print(correlations)

    # 5. Submission
    THRESHOLD = 0.2251
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(load_cached_data=True)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission generation skipped.")


if __name__ == "__main__":
    main()
