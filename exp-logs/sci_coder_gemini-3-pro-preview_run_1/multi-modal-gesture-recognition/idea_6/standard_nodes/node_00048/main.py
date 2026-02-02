import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, _levenshtein_distance, decode_predictions
from library.data_loader import GestureDataset
from library.model import ACGRNet
from library.train import load_dense_labels, Trainer, robust_collate_fn
from library.inference import generate_predictions


def main():
    # 1. Initialization
    set_seed(Config.SEED)

    # 2. Data Preparation
    print("Preparing data...")
    # Load metadata to prepare dense labels cache
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # Cache dense labels for training loss
    cache_path = os.path.join(Config.WORK_DIR, "dense_labels_cache.npy")
    load_dense_labels(combined_df, cache_path, load_cached_data=True)

    # Initialize Datasets
    # We use all data but limit epochs for speed as per "fast baseline" requirement
    train_dataset = GestureDataset(split="train", load_cached_data=True)
    val_dataset = GestureDataset(split="val", load_cached_data=True)

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=robust_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=robust_collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Training
    print("Initializing model and trainer...")
    model = ACGRNet()

    # We pass None for test_loader in Trainer as we handle submission conditionally later
    trainer = Trainer(model, train_loader, val_loader, test_loader=None)

    # Train for a limited number of epochs for a fast baseline
    # 15 epochs is sufficient to see convergence on this dataset size with pre-trained features/embeddings
    # or simple geometric features.
    print("Starting training...")
    trainer.fit(num_epochs=15)

    # 4. Validation & Metric Calculation
    print("Performing final validation...")
    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
    model.eval()
    model.to(Config.DEVICE)

    all_preds = []
    all_truths = []

    # Lists for failure analysis
    sample_errors = []
    feat_lengths = []
    feat_audio_energy = []
    feat_skel_energy = []

    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch["skeleton"].to(Config.DEVICE)
            audio = batch["audio"].to(Config.DEVICE)
            mask = batch["mask"].to(Config.DEVICE)

            # Forward
            logits = model(skeleton, audio, mask)

            # Process batch
            for i in range(logits.size(0)):
                valid_len = batch["lengths"][i].item()

                # Decode Prediction
                seq_logits = logits[i, :valid_len, :]
                pred_seq = decode_predictions(
                    seq_logits, threshold=5, bg_class=Config.BACKGROUND_CLASS_ID
                )
                all_preds.append(pred_seq)

                # Ground Truth
                truth_seq = (
                    batch["seq_labels"][i].tolist()
                    if isinstance(batch["seq_labels"][i], torch.Tensor)
                    else batch["seq_labels"][i]
                )
                # Ensure truth_seq is list of ints
                if isinstance(truth_seq, np.ndarray):
                    truth_seq = truth_seq.tolist()
                all_truths.append(truth_seq)

                # --- Failure Analysis Data Collection ---
                # 1. Error Metric (Levenshtein Distance)
                dist = _levenshtein_distance(pred_seq, truth_seq)
                # Normalize by length of truth (avoid div by zero)
                norm_len = len(truth_seq) if len(truth_seq) > 0 else 1
                # We use raw distance or normalized distance?
                # The metric is Sum(Dist) / Sum(Len).
                # For correlation, error rate per sample is better: Dist / Len.
                error_rate = dist / norm_len
                sample_errors.append(error_rate)

                # 2. Features
                # Length
                feat_lengths.append(valid_len)

                # Audio Energy (Mean absolute amplitude of valid frames)
                # audio shape: (B, T, C)
                # We take the mean over T (valid) and C
                aud_sample = audio[i, :valid_len, :]
                mean_aud = torch.mean(torch.abs(aud_sample)).item()
                feat_audio_energy.append(mean_aud)

                # Skeleton Energy (Mean absolute value - proxy for motion/spread since normalized)
                skel_sample = skeleton[i, :valid_len, :]
                mean_skel = torch.mean(torch.abs(skel_sample)).item()
                feat_skel_energy.append(mean_skel)

    # Compute Final Metric
    # Metric = Total Distance / Total Truth Length
    total_dist = 0
    total_len = 0
    for p, t in zip(all_preds, all_truths):
        total_dist += _levenshtein_distance(p, t)
        total_len += len(t)

    final_metric = total_dist / total_len if total_len > 0 else 0.0

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    if len(sample_errors) > 1:
        # Correlation: Error vs Sequence Length
        corr_len, _ = pearsonr(sample_errors, feat_lengths)
        print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")

        # Correlation: Error vs Audio Energy
        corr_aud, _ = pearsonr(sample_errors, feat_audio_energy)
        print(f"Correlation (Error vs Audio Energy): {corr_aud:.4f}")

        # Correlation: Error vs Skeleton Energy
        corr_skel, _ = pearsonr(sample_errors, feat_skel_energy)
        print(f"Correlation (Error vs Skeleton Energy): {corr_skel:.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    # 6. Submission
    threshold = 0.0824829931972789

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is below threshold ({threshold}). Generating submission..."
        )
        # Use inference module to generate predictions
        generate_predictions(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
