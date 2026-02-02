import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
import library.train_segmentation as train_seg
import library.train_encoder as train_enc
import library.feature_generator as feat_gen
import library.train_aggregator as train_agg
import library.inference as inference
from library.models import DualStreamRNN
from library.data import PatientSequenceDataset
from library.utils import weighted_log_loss_numpy


def perform_failure_analysis(model, device):
    """
    Evaluates the model on the validation set, computes the metric,
    and analyzes correlations between error and metadata.
    """
    print("\nStarting Validation and Failure Analysis...")

    # Load validation dataset
    val_dataset = PatientSequenceDataset(split="val", load_cached_data=True)
    if len(val_dataset) == 0:
        print("Validation dataset is empty.")
        return float("inf")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.STAGE3_BATCH_SIZE,
        shuffle=False,
        collate_fn=PatientSequenceDataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    model.eval()

    all_preds = []
    all_targets = []
    all_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch based on collate_fn return signature
            local_emb, global_ctx, anat_probs, targets, lengths = batch

            local_emb = local_emb.to(device)
            global_ctx = global_ctx.to(device)
            anat_probs = anat_probs.to(device)

            # Forward pass
            logits = model(local_emb, global_ctx, anat_probs)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_lengths.extend(lengths)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 1. Calculate Final Metric
    metric = weighted_log_loss_numpy(all_preds, all_targets)
    print(f"Final Validation Metric: {metric}")

    # 2. Failure Analysis
    # Calculate error magnitude per patient (Mean Binary Cross Entropy)
    # Clip predictions for numerical stability in log
    y_p = np.clip(all_preds, 1e-15, 1 - 1e-15)

    # Binary Cross Entropy per label: -[y*log(p) + (1-y)*log(1-p)]
    bce_matrix = -(all_targets * np.log(y_p) + (1 - all_targets) * np.log(1 - y_p))

    # Mean error per patient across all 8 targets
    patient_error = np.mean(bce_matrix, axis=1)

    # Construct analysis dataframe
    # fracture_count is sum of C1-C7 (indices 0-6)
    fracture_counts = np.sum(all_targets[:, :7], axis=1)

    analysis_df = pd.DataFrame(
        {
            "error": patient_error,
            "num_slices": all_lengths,
            "fracture_count": fracture_counts,
        }
    )

    # Compute Correlations
    # Handle cases with constant values (std=0) to avoid warnings
    if analysis_df["num_slices"].std() > 0:
        corr_slices, _ = pearsonr(analysis_df["error"], analysis_df["num_slices"])
    else:
        corr_slices = 0.0

    if analysis_df["fracture_count"].std() > 0:
        corr_frac, _ = pearsonr(analysis_df["error"], analysis_df["fracture_count"])
    else:
        corr_frac = 0.0

    print("-" * 30)
    print("Failure Analysis Report:")
    print(f"Correlation (Error vs Num Slices): {corr_slices:.4f}")
    print(f"Correlation (Error vs Fracture Count): {corr_frac:.4f}")
    print("-" * 30)

    return metric


def main():
    # 1. Setup and Configuration Overrides
    Config.setup()

    # Optimize for fast baseline execution within 2 hours
    Config.DEBUG = False
    Config.STAGE1_EPOCHS = 2
    Config.STAGE2_EPOCHS = 2
    Config.STAGE3_EPOCHS = 5

    # Ensure efficient batch sizes for A100
    Config.STAGE1_BATCH_SIZE = 32
    Config.STAGE2_BATCH_SIZE = 32
    Config.STAGE3_BATCH_SIZE = 32

    print("Configuration set for fast baseline execution.")

    # 2. Train Stage 1: Segmentation U-Net
    print("\n=== Step 1: Training Segmentation Model ===")
    train_seg.train_stage1(load_cached_data=True)

    # 3. Train Stage 2: Fracture Encoder
    print("\n=== Step 2: Training Fracture Encoder ===")
    train_enc.train_stage2(load_cached_data=True)

    # 4. Feature Generation (Train/Val)
    print("\n=== Step 3: Generating Features for Train/Val ===")
    # We need these features to train Stage 3
    feat_gen.generate_patient_features(splits=["train", "val"], load_cached_data=True)

    # 5. Train Stage 3: Aggregator RNN
    print("\n=== Step 4: Training Sequence Aggregator ===")
    train_agg.train_stage3(load_cached_data=True)

    # 6. Validation and Analysis
    print("\n=== Step 5: Validation & Analysis ===")
    device = torch.device(Config.DEVICE)

    # Load the best Stage 3 model
    model = DualStreamRNN(global_context_dim=512).to(device)
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("Error: Stage 3 checkpoint not found. Validation results may be invalid.")

    val_metric = perform_failure_analysis(model, device)

    # 7. Submission Logic
    # Threshold defined in instructions
    THRESHOLD = 0.9254394427010018

    if val_metric < THRESHOLD:
        print(f"\nMetric {val_metric:.6f} < {THRESHOLD}. Proceeding to Submission.")

        # Generate features for Test set
        print("Generating Test Features...")
        feat_gen.generate_patient_features(splits=["test"], load_cached_data=True)

        # Run Inference
        print("Running Inference...")
        inference.run_inference(load_cached_data=True)
    else:
        print(f"\nMetric {val_metric:.6f} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
