import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config
from library.train import Trainer
from library.utils import set_seed, compute_levenshtein, decode_predictions
from library.data_loader import GestureDataset, CollateFn
from library.model import CGR_GRU


def get_gt_sequence(tensor_labels):
    """
    Extracts sequence of gesture IDs from frame-wise label tensor.
    Collapses duplicates and removes background (0).
    """
    seq = []
    prev = -1
    for x in tensor_labels:
        val = x.item()
        if val != prev:
            if val != 0:
                seq.append(int(val))
            prev = val
    return seq


def main():
    # 1. Setup
    set_seed(Config.SEED)
    os.makedirs("./submission", exist_ok=True)
    device = Config.get_device()

    # 2. Training
    # The Trainer handles data loading, model init, and the training loop
    trainer = Trainer()
    trainer.fit()

    # 3. Final Validation & Failure Analysis
    print("\nRunning Final Validation and Failure Analysis...")

    # Load the best model saved during training
    model = CGR_GRU().to(device)
    checkpoint_path = Config.CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: No checkpoint found. Using current model state.")

    model.eval()

    # Use the validation loader from the trainer
    val_loader = trainer.val_loader

    total_dist = 0.0
    total_gt_gestures = 0

    # Store data for correlation analysis
    analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]
            ids = batch["ids"]

            # Inference
            logits = model(skeleton, audio, lengths=lengths)
            probs = torch.softmax(logits, dim=2)

            # Process each sample in the batch
            for i in range(len(ids)):
                seq_len = lengths[i].item()

                # Slice valid data (remove padding)
                valid_probs = probs[i, :seq_len, :]
                valid_skel = skeleton[i, :seq_len, :]
                valid_audio = audio[i, :seq_len, :]

                # Decode predictions
                pred_seq = decode_predictions(valid_probs)

                # Get Ground Truth
                gt_seq = get_gt_sequence(labels[i, :seq_len])

                # Metric Calculation
                dist = compute_levenshtein(pred_seq, gt_seq)
                n_gt = len(gt_seq)

                total_dist += dist
                total_gt_gestures += n_gt

                # --- Feature Extraction for Failure Analysis ---
                # 1. Error Magnitude (Normalized LER for this sample)
                # If n_gt is 0, we define error as 1.0 if pred is not empty, else 0.0
                if n_gt > 0:
                    sample_error = dist / n_gt
                else:
                    sample_error = 1.0 if len(pred_seq) > 0 else 0.0

                # 2. Sequence Length
                feat_len = seq_len

                # 3. Motion Energy: Mean of absolute temporal differences of skeleton
                # skel shape: (T, 60)
                if seq_len > 1:
                    skel_diff = valid_skel[1:] - valid_skel[:-1]
                    feat_motion = torch.mean(torch.abs(skel_diff)).item()
                else:
                    feat_motion = 0.0

                # 4. Audio Energy: Std dev of audio features
                feat_audio = torch.std(valid_audio).item()

                analysis_data.append(
                    {
                        "error": sample_error,
                        "length": feat_len,
                        "motion": feat_motion,
                        "audio_energy": feat_audio,
                    }
                )

    # Compute Final Metric
    final_metric = total_dist / total_gt_gestures if total_gt_gestures > 0 else 1.0
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlations
    if len(analysis_data) > 1:
        df_analysis = pd.DataFrame(analysis_data)

        # Correlation: Error vs Length
        corr_len, _ = pearsonr(df_analysis["error"], df_analysis["length"])
        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")

        # Correlation: Error vs Motion
        corr_motion, _ = pearsonr(df_analysis["error"], df_analysis["motion"])
        print(f"Correlation (Error vs Motion Energy): {corr_motion:.4f}")

        # Correlation: Error vs Audio
        corr_audio, _ = pearsonr(df_analysis["error"], df_analysis["audio_energy"])
        print(f"Correlation (Error vs Audio Energy): {corr_audio:.4f}")

    # 4. Submission
    THRESHOLD = 0.0824829931972789

    if final_metric < THRESHOLD:
        print(f"Metric passed threshold ({THRESHOLD}). Generating submission...")

        # Load Test Data
        test_dataset = GestureDataset(split="test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=CollateFn(mode="test"),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        submission_data = []

        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue

                skeleton = batch["skeleton"].to(device)
                audio = batch["audio"].to(device)
                lengths = batch["lengths"]
                ids = batch["ids"]

                logits = model(skeleton, audio, lengths=lengths)
                probs = torch.softmax(logits, dim=2)

                for i in range(len(ids)):
                    seq_len = lengths[i].item()
                    valid_probs = probs[i, :seq_len, :]

                    pred_seq = decode_predictions(valid_probs)

                    # Convert ID: 'Sample00300' -> 300
                    # Cite debug_lesson_3: Strictly Align Submission IDs
                    try:
                        sample_id = int(str(ids[i]).replace("Sample", ""))
                    except ValueError:
                        sample_id = ids[i]

                    # Format: Space-separated sequence
                    # Cite debug_lesson_2: Encapsulate Variable-Length Sequences
                    pred_str = " ".join(map(str, pred_seq))

                    submission_data.append({"Id": sample_id, "Sequence": pred_str})

        # Save to file using pandas
        sub_path = "./submission/submission.csv"
        df_sub = pd.DataFrame(submission_data)
        df_sub = df_sub[["Id", "Sequence"]]
        df_sub.to_csv(sub_path, index=False)

        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"Metric {final_metric} did not pass threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
