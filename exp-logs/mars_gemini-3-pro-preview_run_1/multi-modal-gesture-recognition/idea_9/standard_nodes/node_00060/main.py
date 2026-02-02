import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import medfilt

# Import provided library modules
from library import config, utils, data_loader, model, trainer


def main():
    # 1. Setup and Initialization
    # Ensure reproducibility
    utils.set_seed()

    print("Initializing ID-GFN Training Pipeline...")

    # Initialize the Trainer from the library
    # This sets up the model, optimizer, loss, and scheduler
    t = trainer.Trainer()

    # 2. Training
    # We use the fit method which handles the training loop, validation, and checkpointing.
    # Given the dataset size (approx 300 sequences), we use the full dataset to ensure
    # the model learns the 20 classes effectively within the 50 epochs defined in config.
    # This fits within the "fast baseline" constraint due to the small data volume.
    print("Starting Training...")
    t.fit()

    # 3. Validation Assessment & Failure Analysis
    print("\nStarting Validation Assessment and Failure Analysis...")

    # Load the best model saved during training
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    t.model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    t.model.eval()

    # Create Validation DataLoader
    val_dataset = data_loader.GestureDataset(config.VAL_METADATA_PATH, mode="val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=data_loader.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Containers for analysis
    all_preds_seq = []
    all_targets_seq = []
    sample_metrics = []

    with torch.no_grad():
        for skeletons, audios, labels, mask in val_loader:
            skeletons = skeletons.to(config.DEVICE)
            audios = audios.to(config.DEVICE)
            # labels are needed for ground truth comparison
            labels = labels.to(config.DEVICE)
            lengths = mask.sum(dim=1).cpu()

            # Inference
            logits = t.model(skeletons, audios, lengths)
            preds = torch.argmax(logits, dim=2).cpu().numpy()
            labels_np = labels.cpu().numpy()

            batch_size = preds.shape[0]
            for i in range(batch_size):
                length = int(lengths[i].item())

                # Extract valid sequence
                p_seq = preds[i, :length]
                t_seq = labels_np[i, :length]

                # Post-processing: Median Filter
                if len(p_seq) >= 5:
                    p_seq = medfilt(p_seq, kernel_size=5)

                # Decode to Gesture IDs
                pred_decoded = utils.rle_decode(
                    p_seq, background_label=config.BACKGROUND_LABEL, min_duration=5
                )

                target_decoded = utils.rle_decode(
                    t_seq,
                    background_label=config.BACKGROUND_LABEL,
                    min_duration=1,  # Keep short ground truth
                )

                # Compute Metric for this sample
                dist = utils.levenshtein_distance(pred_decoded, target_decoded)

                all_preds_seq.append(pred_decoded)
                all_targets_seq.append(target_decoded)

                # Store stats for correlation analysis
                sample_metrics.append(
                    {
                        "seq_len": length,
                        "num_gestures": len(target_decoded),
                        "distance": dist,
                    }
                )

    # Compute Final Validation Metric (Total Distance / Total Gestures)
    total_distance = sum(m["distance"] for m in sample_metrics)
    total_gestures = sum(m["num_gestures"] for m in sample_metrics)

    final_metric = total_distance / total_gestures if total_gestures > 0 else 0.0

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    if sample_metrics:
        df_analysis = pd.DataFrame(sample_metrics)

        # Correlation: Error Magnitude (Distance) vs Sequence Length
        corr_len, _ = pearsonr(df_analysis["seq_len"], df_analysis["distance"])

        # Correlation: Error Magnitude (Distance) vs Number of Gestures
        corr_num, _ = pearsonr(df_analysis["num_gestures"], df_analysis["distance"])

        print(
            "\nFailure Analysis - Correlations with Error Magnitude (Levenshtein Distance):"
        )
        print(f"Correlation with Sequence Length: {corr_len:.4f}")
        print(f"Correlation with Number of Gestures: {corr_num:.4f}")

    # 4. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.0824829931972789

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        t.predict()
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
