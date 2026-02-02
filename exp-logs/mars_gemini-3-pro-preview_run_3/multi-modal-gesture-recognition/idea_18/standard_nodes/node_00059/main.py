import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import nltk

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.utils import set_seed, compute_levenshtein
from library.trainer import Trainer
from library.model import validate
from library.inference import predict_sliding_window, decode_predictions


def main():
    # 1. Setup
    set_seed(42)

    # 2. Initialize Trainer
    # We use 25 epochs to ensure the run completes quickly (fast baseline)
    # while still allowing convergence for the recurrent model.
    trainer = Trainer(
        base_dir="./",
        cache_dir="./working/idea_18",
        submission_dir="./submission",
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=25,
        patience=8,
    )

    # 3. Load Data & Setup Model
    print("Loading Data...")
    trainer.load_data()

    print("Setting up Model...")
    trainer.setup_model()

    # 4. Train
    print("Starting Training...")
    trainer.train()

    # 5. Final Evaluation on Validation Set
    # Load the best model weights saved during training to ensure we evaluate the best state
    if os.path.exists(trainer.best_model_path):
        trainer.model.load_state_dict(torch.load(trainer.best_model_path))
    else:
        print("Warning: Best model not found. Using current weights.")

    # Compute Final Metric using the validate function
    # This computes the normalized Levenshtein score (Error Rate)
    final_metric = validate(
        trainer.model, trainer.val_loader, trainer.device, trainer.val_targets
    )
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    trainer.model.eval()

    # We need to run inference on the validation set to get specific predictions for analysis
    # (validate() returns the aggregate score, but we need per-sample errors)
    val_probs, val_counts = predict_sliding_window(
        trainer.model, trainer.val_loader, trainer.device
    )
    val_preds = decode_predictions(val_probs, val_counts)

    analysis_data = []

    for sample in trainer.val_loader.dataset.samples:
        sid = sample["sample_id"]
        target_seq = trainer.val_targets.get(sid, [])
        pred_seq = val_preds.get(sid, [])

        # Calculate Error for this sample
        error = compute_levenshtein(pred_seq, target_seq)

        # Extract Features
        # Feature 1: Sequence Length (number of frames)
        seq_len = sample["skeleton"].shape[0]

        # Feature 2: Complexity (Number of gestures in ground truth)
        num_gestures = len(target_seq)

        # Feature 3: Average Motion Energy (Mean velocity magnitude)
        skel = sample["skeleton"]  # (T, 20, 3)
        if skel.shape[0] > 1:
            vel = skel[1:] - skel[:-1]
            vel_mag = np.linalg.norm(vel, axis=2).mean()  # Mean over joints and time
        else:
            vel_mag = 0.0

        analysis_data.append(
            {
                "error": error,
                "seq_len": seq_len,
                "num_gestures": num_gestures,
                "avg_motion": vel_mag,
            }
        )

    df_analysis = pd.DataFrame(analysis_data)

    # Compute and Print Correlations
    if not df_analysis.empty and len(df_analysis) > 1:
        corr_len = df_analysis["error"].corr(df_analysis["seq_len"])
        corr_num = df_analysis["error"].corr(df_analysis["num_gestures"])
        corr_mot = df_analysis["error"].corr(df_analysis["avg_motion"])

        print(f"Correlation (Error vs Seq Length): {corr_len}")
        print(f"Correlation (Error vs Num Gestures): {corr_num}")
        print(f"Correlation (Error vs Avg Motion): {corr_mot}")
    else:
        print("Insufficient data for failure analysis.")

    # 7. Conditional Submission
    # Only generate submission if metric is better (lower) than the threshold
    submission_threshold = 0.2251
    if final_metric < submission_threshold:
        print(
            f"Metric {final_metric} < {submission_threshold}. Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(f"Metric {final_metric} >= {submission_threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
