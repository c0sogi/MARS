import sys
import os
import torch
import numpy as np
import pandas as pd
import itertools
import nltk
from scipy.stats import pearsonr

# Ensure library modules are importable
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.trainer import Trainer
from library.inference import (
    preprocess_sequence,
    sliding_window_inference,
    decode_predictions,
)


def analyze_failures(trainer):
    """
    Performs failure analysis on the validation set to identify error correlations.
    """
    print("Running Failure Analysis on Validation Set...")

    # Ensure best model is loaded
    device = trainer.device
    if os.path.exists(Config.MODEL_SAVE_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        )
    trainer.model.eval()

    val_dataset = trainer.val_loader.dataset
    stats = trainer.stats

    results = []

    # Iterate over validation samples
    for i in range(len(val_dataset.ids_list)):
        # Extract Data
        sample_id = val_dataset.ids_list[i]
        audio = val_dataset.audio_list[i]
        skel = val_dataset.skeleton_list[i]
        gt_labels_frame = val_dataset.labels_list[i]

        # 1. Inference
        features = preprocess_sequence(audio, skel, stats)
        probs = sliding_window_inference(trainer.model, features, device)
        pred_seq = decode_predictions(probs)

        # 2. Ground Truth Sequence
        gt_rle = [(k, len(list(g))) for k, g in itertools.groupby(gt_labels_frame)]
        gt_seq = [int(k) for k, d in gt_rle if k != Config.BACKGROUND_LABEL]

        # 3. Compute Error (Levenshtein)
        dist = nltk.edit_distance(pred_seq, gt_seq)

        # 4. Extract Meta-Features
        seq_len = features.shape[0]
        num_gt_gestures = len(gt_seq)

        # Audio Energy (Mean of absolute normalized features)
        # features[:, :13] are audio MFCCs
        audio_energy = torch.mean(torch.abs(features[:, : Config.N_MFCC])).item()

        # Kinematic Energy (Mean of absolute velocity/accel)
        # features[:, 13:] are skeletal features.
        # Structure is [Pos, Vel, Acc]. Vel starts at 13 + 60 = 73?
        # SKELETON_INPUT_DIM = 180. Pos(60), Vel(60), Acc(60).
        # Indices: Pos [13:73], Vel [73:133], Acc [133:193]
        # Let's just take mean of the kinematic part generally as "motion intensity"
        kinematic_intensity = torch.mean(torch.abs(features[:, Config.N_MFCC :])).item()

        results.append(
            {
                "sample_id": sample_id,
                "error": dist,
                "seq_len": seq_len,
                "num_gestures": num_gt_gestures,
                "audio_energy": audio_energy,
                "kinematic_intensity": kinematic_intensity,
            }
        )

    df = pd.DataFrame(results)

    if len(df) > 0:
        print("\nCorrelation between Error (Levenshtein Distance) and Input Features:")
        features_to_corr = [
            "seq_len",
            "num_gestures",
            "audio_energy",
            "kinematic_intensity",
        ]

        for feat in features_to_corr:
            if feat in df.columns:
                # Handle constant columns to avoid warnings
                if df[feat].std() == 0:
                    corr = 0.0
                else:
                    corr, _ = pearsonr(df["error"], df[feat])
                print(f"  Error vs {feat}: {corr:.4f}")
    else:
        print("No validation samples found for analysis.")


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    print("initializing configuration...")
    # Override Config for faster execution within time limits
    Config.NUM_EPOCHS = 15
    Config.PATIENCE = 5

    # Ensure reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Training
    # ==========================================
    print("Initializing Trainer...")
    # This will load data (using cache if available)
    trainer = Trainer(config=Config)

    print("Starting Training Loop...")
    trainer.train()

    # ==========================================
    # 3. Metric Reporting
    # ==========================================
    final_metric = trainer.best_lev_dist
    # Print exact format required
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    try:
        analyze_failures(trainer)
    except Exception as e:
        print(f"Warning: Failure analysis failed with error: {e}")

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.2251

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) is NOT below threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
