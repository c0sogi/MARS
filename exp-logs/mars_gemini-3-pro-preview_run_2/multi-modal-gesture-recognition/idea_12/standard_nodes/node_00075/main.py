import os
import torch
import numpy as np
import pandas as pd
import scipy.stats
from nltk import edit_distance

from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.trainer import Trainer


def main():
    # 1. Setup & Initialization
    set_seed(Config.SEED)
    Config.init_dirs()

    # 2. Model Training
    # Initialize Trainer (loads cached data automatically)
    trainer = Trainer(load_cached_data=True)

    # Execute training
    # We use the config's epoch setting. The dataset size allows this to complete
    # within the time limit on the provided hardware.
    trainer.train(num_epochs=Config.NUM_EPOCHS)

    # 3. Validation Assessment
    # Load the best model checkpoint saved during training
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        trainer.model.load_state_dict(
            torch.load(checkpoint_path, map_location=trainer.device)
        )
        print("Best model checkpoint loaded.")
    else:
        print("Warning: Checkpoint not found. Using current model state.")

    trainer.model.eval()

    val_loader = trainer.val_loader
    all_preds = []
    all_targets = []

    # Containers for failure analysis
    sample_errors = []
    feature_stats = []

    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(trainer.device)
            mask = batch["mask"].to(trainer.device)
            labels_cls = batch["labels_cls"].to(trainer.device)

            # Forward pass
            outputs = trainer.model(features, mask)
            # Use Stage 3 outputs for final prediction
            s3_probs = outputs["stage3"]["cls_probs"]

            # Decode predictions
            batch_preds = trainer.decode_predictions(s3_probs, mask)

            # Prepare data for analysis
            labels_cls_np = labels_cls.cpu().numpy()
            mask_np = mask.cpu().numpy()
            features_np = features.cpu().numpy()

            for i in range(len(batch_preds)):
                valid_len = int(mask_np[i].sum())

                # Reconstruct ground truth sequence from frame-wise labels
                t_seq_raw = labels_cls_np[i, :valid_len]
                t_seq = []
                prev = -1
                for l in t_seq_raw:
                    if l != prev:
                        if l != 0:  # 0 is background
                            t_seq.append(int(l))
                        prev = l

                all_preds.append(batch_preds[i])
                all_targets.append(t_seq)

                # --- Failure Analysis Calculation ---
                # Compute Levenshtein distance for this specific sample
                dist = edit_distance(batch_preds[i], t_seq)

                # Normalize error rate for correlation analysis
                if len(t_seq) > 0:
                    norm_err = dist / len(t_seq)
                else:
                    norm_err = 0.0 if dist == 0 else 1.0
                sample_errors.append(norm_err)

                # Extract Input Features
                # Feature Mapping: [Pos (0-35), Vel (36-71), Audio (72-84)]
                feat_valid = features_np[i, :valid_len, :]

                # Calculate Mean Velocity Magnitude
                vel_data = feat_valid[:, 36:72]
                mean_vel = np.mean(np.abs(vel_data))

                feature_stats.append({"length": valid_len, "mean_velocity": mean_vel})

    # Compute and Print Final Metric
    final_metric = compute_levenshtein(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = pd.DataFrame(feature_stats)
    df_analysis["error"] = sample_errors

    # Compute correlations
    for col in ["length", "mean_velocity"]:
        if col in df_analysis.columns:
            # Check for variance to avoid warnings
            if df_analysis[col].std() > 0 and df_analysis["error"].std() > 0:
                corr, _ = scipy.stats.pearsonr(df_analysis[col], df_analysis["error"])
                print(f"Correlation between {col} and error: {corr}")
            else:
                print(
                    f"Correlation between {col} and error: Undefined (insufficient variance)"
                )

    # 5. Submission Generation
    threshold = 0.10854816824966079
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) meets threshold (< {threshold}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nMetric ({final_metric}) does not meet threshold (< {threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
