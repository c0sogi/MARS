import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import weighted_log_loss
from library.segmentation_engine import train_segmenter, generate_dataset_masks
from library.encoder_engine import train_encoder, extract_patient_features
from library.sequence_engine import train_aggregator, predict_pipeline
from library.models import AttentionalRNN
from library.datasets import FeatureSequenceDataset


def main():
    # 1. Setup
    Config.setup()

    # Load Metadata
    if (
        not os.path.exists(Config.TRAIN_METADATA_PATH)
        or not os.path.exists(Config.VAL_METADATA_PATH)
        or not os.path.exists(Config.TEST_METADATA_PATH)
    ):
        print("Error: Metadata files not found.")
        return

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Combine all metadata for batch processing steps (Masks & Features)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("Metadata loaded.")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Stage 1: Segmentation
    # Train the U-Net Localizer
    train_segmenter(train_df, val_df)

    # Generate Masks for the entire dataset (Train + Val + Test)
    # This uses the trained model to create .npy mask files
    mask_dir = generate_dataset_masks(all_df, load_cached_data=True)

    # 3. Stage 2: Slice Encoder
    # Train the 2.5D CNN Encoder
    train_encoder(train_df, val_df, mask_dir)

    # Extract Features for the entire dataset
    # This uses the trained encoder to create .npy feature sequence files
    feature_dir = extract_patient_features(all_df, mask_dir, load_cached_data=True)

    # 4. Stage 3: Sequence Aggregator
    # Train the RNN Aggregator
    train_aggregator(train_df, val_df, feature_dir)

    # 5. Validation & Metric Calculation
    print("\nRunning Final Validation...")
    device = Config.DEVICE

    # Load the trained aggregator model
    model = AttentionalRNN(
        input_dim=Config.ENCODER_HIDDEN_DIM,
        hidden_dim=Config.SEQ_HIDDEN_DIM,
        num_layers=Config.SEQ_NUM_LAYERS,
    ).to(device)

    weights_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Warning: Aggregator checkpoint not found for validation.")

    model.eval()

    # Prepare Validation Loader
    val_dataset = FeatureSequenceDataset(val_df, feature_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.SEQ_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []

    # Inference Loop (No Grad)
    with torch.no_grad():
        for features, masks, labels in val_loader:
            features = features.to(device)
            masks = masks.to(device)

            logits = model(features, masks)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
    else:
        all_preds = np.zeros((0, 8))
        all_targets = np.zeros((0, 8))

    # Calculate Metric
    val_metric = weighted_log_loss(all_targets, all_preds)
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate weighted log loss per sample (row)
    # Weights shape: (1, 8)
    weights = np.array(
        [Config.LOSS_WEIGHTS.get(col, 1.0) for col in Config.TARGET_COLS]
    ).reshape(1, -1)

    # Clip predictions for stability
    epsilon = 1e-15
    y_pred = np.clip(all_preds, epsilon, 1 - epsilon)
    y_true = all_targets

    # Compute loss matrix: (N, 8)
    loss_matrix = -weights * (
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

    # Average loss per patient (across classes)
    patient_errors = np.mean(loss_matrix, axis=1)

    # Create analysis dataframe
    analysis_df = val_df.copy().reset_index(drop=True)
    analysis_df["error"] = patient_errors

    # Feature 1: Sequence Length
    # Retrieve sequence length from feature files
    seq_lens = []
    for uid in analysis_df["StudyInstanceUID"]:
        fpath = os.path.join(feature_dir, f"{uid}.npy")
        if os.path.exists(fpath):
            arr = np.load(fpath)
            seq_lens.append(len(arr))
        else:
            seq_lens.append(0)
    analysis_df["seq_len"] = seq_lens

    # Feature 2: Has Bounding Box (Proxy for difficulty/fracture presence)
    analysis_df["has_bbox_int"] = analysis_df["has_bounding_box"].astype(int)

    # Correlations
    if len(analysis_df) > 1:
        corr_len = analysis_df["error"].corr(analysis_df["seq_len"])
        corr_bbox = analysis_df["error"].corr(analysis_df["has_bbox_int"])
        print(f"Correlation (Error vs Seq Len): {corr_len}")
        print(f"Correlation (Error vs Has BBox): {corr_bbox}")
    else:
        print("Not enough validation samples for correlation analysis.")

    # 7. Submission
    THRESHOLD = 0.9440845186799401

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric {val_metric} passed threshold {THRESHOLD}. Generating submission..."
        )
        predict_pipeline(test_df, feature_dir)
    else:
        print(
            f"\nValidation metric {val_metric} did not pass threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
