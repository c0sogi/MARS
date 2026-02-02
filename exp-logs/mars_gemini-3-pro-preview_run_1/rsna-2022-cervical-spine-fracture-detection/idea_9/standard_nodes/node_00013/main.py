import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.engine import (
    Stage1Trainer,
    Stage2Trainer,
    Stage3Trainer,
    SubmissionGenerator,
    FeatureExtractor,
)
from library.utils import WeightedLogLoss
from library.data import SequenceDataset
from library.models import HierarchicalAggregator


def main():
    # 1. Setup and Configuration
    Config.setup()

    # Override epochs for Fast Baseline execution (Time Limit: 2 hours)
    # Reducing epochs to ensure completion while demonstrating functionality.
    Config.SEG_EPOCHS = 1
    Config.ENC_EPOCHS = 1
    Config.RNN_EPOCHS = 3

    # Ensure batch sizes are safe for the provided hardware
    Config.SEG_BATCH_SIZE = 32
    Config.ENC_BATCH_SIZE = 32
    Config.RNN_BATCH_SIZE = 4

    print("=== Starting DS-HRN Pipeline ===")

    # 2. Train Stage 1: Anatomical Localizer (U-Net)
    print("\n[Step 1/5] Training Stage 1: Anatomical Localizer...")
    s1_trainer = Stage1Trainer()
    s1_trainer.train()

    # 3. Train Stage 2: Dual-Branch Encoder
    print("\n[Step 2/5] Training Stage 2: Dual-Branch Encoder...")
    s2_trainer = Stage2Trainer()
    s2_trainer.train()

    # 4. Train Stage 3: Hierarchical Aggregator (Bi-GRU)
    # This step implicitly runs Feature Extraction for Train/Val sets
    print("\n[Step 3/5] Training Stage 3: Hierarchical Aggregator...")
    s3_trainer = Stage3Trainer()
    s3_trainer.train()

    # 5. Validation and Metrics
    print("\n[Step 4/5] Performing Validation...")

    # Load Validation Data
    val_meta_path = Config.VAL_METADATA_PATH
    val_df = pd.read_csv(val_meta_path)
    feature_dir = os.path.join(Config.CACHE_DIR, "features")

    # Use SequenceDataset to load pre-computed features
    val_ds = SequenceDataset(val_df, feature_dir, mode="val")

    def collate_fn(batch):
        features, targets, ids = zip(*batch)
        features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        targets_stacked = torch.stack(targets)
        return features_padded, targets_stacked, ids

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.RNN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load Model
    device = Config.DEVICE
    model = HierarchicalAggregator().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    criterion = WeightedLogLoss()

    all_targets = []
    all_logits = []

    # Inference loop
    with torch.no_grad():
        for features, targets, _ in val_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Concatenate
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute Metric
    # WeightedLogLoss returns the mean weighted loss
    val_loss = criterion(all_logits, all_targets).item()

    print(f"Final Validation Metric: {val_loss:.16f}")

    # 6. Failure Analysis
    print("\n[Failure Analysis]")

    # Calculate loss per patient (row) to correlate
    # We manually compute the loss per sample to analyze distribution
    # Weights: [1,1,1,1,1,1,1,7]
    weights = torch.tensor([1.0] * 7 + [7.0])

    # BCE per element (unreduced)
    bce_none = torch.nn.functional.binary_cross_entropy_with_logits(
        all_logits, all_targets, reduction="none"
    )

    # Apply weights
    weighted_bce = bce_none * weights

    # Average across the 8 classes for each patient to get a "patient-level error score"
    # Note: The competition metric averages across ALL rows (patients * classes).
    # Here we aggregate per patient to see which patients are hard.
    patient_errors = weighted_bce.mean(dim=1).numpy()

    # Get patient_overall labels
    patient_labels = all_targets[:, 7].numpy()

    # Correlation
    correlation = np.corrcoef(patient_errors, patient_labels)[0, 1]

    print(f"Correlation between Error and 'patient_overall' label: {correlation:.4f}")
    print(
        "Interpretation: Positive correlation implies higher error on fracture cases (False Negatives)."
    )
    print(
        "                Negative correlation implies higher error on healthy cases (False Positives)."
    )

    # 7. Submission
    THRESHOLD = 0.9254394427010018

    if val_loss < THRESHOLD:
        print(
            f"\n[Step 5/5] Validation metric ({val_loss:.6f}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generator = SubmissionGenerator()
        generator.generate()
    else:
        print(
            f"\n[Step 5/5] Validation metric ({val_loss:.6f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
