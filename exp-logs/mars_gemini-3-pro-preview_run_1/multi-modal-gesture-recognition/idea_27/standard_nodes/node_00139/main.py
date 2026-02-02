import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import nltk
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed, decode_predictions
from library.data_loader import get_dataloaders
from library.model import MPCNet
from library.train import train_epoch, validate


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Configure for Fast Baseline
    # Reduce epochs to ensure quick execution within time limits
    Config.NUM_EPOCHS = 20

    # 3. Data Loading
    # Load full datasets; the dataset size is small enough for a fast baseline without subsampling
    train_loader, val_loader, test_loader = get_dataloaders()

    # 4. Model Initialization
    model = MPCNet().to(device)

    # 5. Optimization Setup
    # Class Weights: 0.5 for Background, 1.0 for others
    weights = torch.ones(Config.NUM_CLASSES).to(device)
    weights[Config.BACKGROUND_CLASS_ID] = Config.BACKGROUND_WEIGHT_VALUE

    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=Config.LABEL_SMOOTHING, reduction="mean"
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # 6. Training Loop
    best_ler = float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_ler = validate(model, val_loader, device)
        scheduler.step()

        # Save best model
        if val_ler < best_ler:
            best_ler = val_ler
            torch.save(model.state_dict(), best_model_path)

    # 7. Final Metric Reporting
    print(f"Final Validation Metric: {best_ler}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    errors = []
    feat_lens = []
    feat_audio_energy = []
    feat_skel_motion = []

    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            length = batch["length"].to(device)

            logits = model(skeleton, audio, length)
            probs = torch.softmax(logits, dim=2)

            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            lengths_np = length.cpu().numpy()
            skel_np = skeleton.cpu().numpy()
            audio_np = audio.cpu().numpy()

            for i in range(len(probs_np)):
                valid_len = lengths_np[i]

                # Decode Prediction
                # Slice valid probabilities
                sample_probs = probs_np[i, :valid_len, :]
                pred_seq = decode_predictions(sample_probs)

                # Extract Ground Truth Sequence
                valid_labels = labels_np[i, :valid_len]
                gt_seq = []
                if len(valid_labels) > 0:
                    curr = valid_labels[0]
                    if curr != Config.BACKGROUND_CLASS_ID:
                        gt_seq.append(curr)
                    for lbl in valid_labels[1:]:
                        if lbl != curr:
                            curr = lbl
                            if curr != Config.BACKGROUND_CLASS_ID:
                                gt_seq.append(curr)

                # Compute Sample-wise Error Rate
                dist = nltk.edit_distance(pred_seq, gt_seq)
                if len(gt_seq) > 0:
                    err = dist / len(gt_seq)
                else:
                    err = 0.0 if dist == 0 else 1.0

                errors.append(err)
                feat_lens.append(valid_len)

                # Feature: Audio Energy (Mean Absolute Amplitude)
                if valid_len > 0:
                    feat_audio_energy.append(
                        np.mean(np.abs(audio_np[i, :valid_len, :]))
                    )
                else:
                    feat_audio_energy.append(0.0)

                # Feature: Skeleton Motion (Mean Frame-to-Frame Difference)
                if valid_len > 1:
                    s = skel_np[i, :valid_len, :]
                    motion = np.mean(np.abs(np.diff(s, axis=0)))
                    feat_skel_motion.append(motion)
                else:
                    feat_skel_motion.append(0.0)

    # Compute Correlations
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "length": feat_lens,
            "audio_energy": feat_audio_energy,
            "skel_motion": feat_skel_motion,
        }
    )

    print("Correlation between Error Magnitude and Input Features:")
    for col in ["length", "audio_energy", "skel_motion"]:
        # Handle constant columns to avoid warnings
        if df_analysis[col].std() == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(df_analysis["error"], df_analysis[col])
        print(f"  {col}: {corr:.4f}")

    # 9. Submission Generation
    THRESHOLD = 0.05697278911564626

    if best_ler < THRESHOLD:
        print("\nValidation metric meets threshold. Generating submission...")

        test_df = test_loader.dataset.df
        results = []
        current_idx = 0

        with torch.no_grad():
            for batch in test_loader:
                skeleton = batch["skeleton"].to(device)
                audio = batch["audio"].to(device)
                length = batch["length"].to(device)

                logits = model(skeleton, audio, length)
                probs = torch.softmax(logits, dim=2)

                probs_np = probs.cpu().numpy()
                lengths_np = length.cpu().numpy()
                batch_size = len(probs_np)

                # Map predictions back to Sample IDs
                # collate_fn sorts batch by length descending.
                # We must apply the same sort to the corresponding dataframe slice.
                df_slice = test_df.iloc[current_idx : current_idx + batch_size].copy()
                df_slice["sort_len"] = df_slice["num_frames"]
                df_slice_sorted = df_slice.sort_values(
                    by="sort_len", ascending=False, kind="mergesort"
                )

                for i in range(batch_size):
                    valid_len = lengths_np[i]
                    sample_probs = probs_np[i, :valid_len, :]
                    pred_seq = decode_predictions(sample_probs)

                    # Format: 2,12,3
                    pred_str = ",".join(map(str, pred_seq))

                    sample_id = df_slice_sorted.iloc[i]["sample_id"]
                    results.append((sample_id, pred_str))

                current_idx += batch_size

        # Save Submission
        with open(Config.SUBMISSION_PATH, "w") as f:
            for sid, pred in results:
                f.write(f"{sid},{pred}\n")

        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nValidation metric {best_ler} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
