import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# Monkey patch tqdm to suppress output as per strict requirements
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# Import library modules
from library.config import Config

# Configuration Overrides for Fast Baseline
# We enable DEBUG mode to restrict the dataset size, ensuring the pipeline
# completes within the 2-hour limit. 200,000 samples provide a reasonable
# trade-off between speed and representativeness.
Config.DEBUG = True
Config.DEBUG_SUBSET_SIZE = 200000
Config.NUM_EPOCHS = 15  # Sufficient for convergence on this subset size
Config.BATCH_SIZE = 2048

from library.utils import HierarchyMapper
from library.feature_extractor import extract_and_save_features
from library.datasets import FeatureDataset
from library.trainer import Trainer
from library.model import generate_submission


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis by correlating prediction errors with feature vector norms.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_feature_norms = []

    with torch.no_grad():
        for features, l1, l2, l3 in val_loader:
            features = features.to(device)
            l3 = l3.to(device)

            # Forward pass
            _, _, p3 = model(features)
            _, predicted = torch.max(p3, 1)

            all_preds.append(predicted.cpu().numpy())
            all_targets.append(l3.cpu().numpy())

            # Calculate L2 norm of feature vectors
            # This serves as a proxy for "signal strength" or image complexity
            norms = torch.norm(features, p=2, dim=1)
            all_feature_norms.append(norms.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_feature_norms = np.concatenate(all_feature_norms)

    # Error vector: 1 if incorrect, 0 if correct
    errors = (all_preds != all_targets).astype(int)

    # Calculate correlation
    if len(errors) > 1:
        correlation = np.corrcoef(errors, all_feature_norms)[0, 1]
    else:
        correlation = 0.0

    print(
        f"Failure Analysis - Correlation between Error and Feature Norm: {correlation:.6f}"
    )


def main():
    # Set seed for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # 1. Feature Extraction
    # Extracts features from raw BSON files if not already cached.
    # Uses the subset size defined in Config.
    print("Starting Feature Extraction...")
    extract_and_save_features(
        load_cached_data=True, subset_size=Config.DEBUG_SUBSET_SIZE
    )

    # 2. Data Preparation
    print("Initializing Datasets...")
    mapper = HierarchyMapper()
    mapper.process()

    # Train Dataset
    train_dataset = FeatureDataset(
        features_path=Config.TRAIN_FEATURES,
        labels_path=Config.TRAIN_LABELS,
        hierarchy_mapper=mapper,
        mode="train",
        subset_size=Config.DEBUG_SUBSET_SIZE,
    )

    # Validation Dataset
    val_dataset = FeatureDataset(
        features_path=Config.VAL_FEATURES,
        labels_path=Config.VAL_LABELS,
        hierarchy_mapper=mapper,
        mode="val",
        subset_size=Config.DEBUG_SUBSET_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Training
    print("Starting Training...")
    trainer = Trainer()
    model = trainer.fit(train_loader, val_loader)

    # 4. Final Validation
    print("Performing Final Validation...")
    val_loss, val_acc = trainer.evaluate(val_loader)
    # Print exactly as requested
    print(f"Final Validation Metric: {val_acc}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    analyze_failures(model, val_loader, trainer.device)

    # 6. Submission
    # Threshold defined in task description
    THRESHOLD = 0.6239621493939094

    if val_acc > THRESHOLD:
        print(
            f"Validation metric {val_acc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Load Test Dataset
        # Note: In DEBUG mode, this will load the subset of test features extracted.
        test_dataset = FeatureDataset(
            features_path=Config.TEST_FEATURES,
            ids_path=Config.TEST_IDS,
            mode="test",
            subset_size=Config.DEBUG_SUBSET_SIZE,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Generate and Save
        generate_submission(model, test_loader, mapper)
    else:
        print(
            f"Validation metric {val_acc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
