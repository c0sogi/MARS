import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
import library.config as config
from library.feature_extractor import DualBackbone, extract_dataset
from library.hierarchy_utils import HierarchyMapper
from library.dataset import CachedFeatureDataset
from library.model import HierarchicalMLP
import library.trainer as trainer
import library.inference as inference


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def custom_feature_extraction():
    """
    Orchestrates feature extraction with specific sample sizes:
    - Train: Subset (200k) for speed.
    - Val: Full for accurate metrics.
    - Test: Full for submission.
    """
    # Check if files exist to avoid redundant computation
    train_exists = os.path.exists(config.TRAIN_FEATURES)
    val_exists = os.path.exists(config.VAL_FEATURES)
    test_exists = os.path.exists(config.TEST_FEATURES)

    if train_exists and val_exists and test_exists:
        print("Features already cached. Skipping extraction.")
        return

    print("Starting custom feature extraction pipeline...")

    # Initialize shared resources
    mapper = HierarchyMapper(load_cached_data=True)
    model = DualBackbone().to(config.DEVICE)
    model.eval()

    # 1. Extract Training Features (Subset)
    if not train_exists:
        print("Extracting Training Features (Subset: 200,000)...")
        # Temporarily set config to limit processing
        config.DEBUG_SAMPLE_SIZE = 200000
        feats, ids, labels = extract_dataset(
            config.TRAIN_META,
            config.TRAIN_BSON,
            model,
            mapper,
            is_test=False,
            desc="Training Set",
        )
        np.save(config.TRAIN_FEATURES, feats)
        np.save(config.TRAIN_IDS, ids)
        np.save(config.TRAIN_LABELS_L3, labels)
        del feats, ids, labels

    # 2. Extract Validation Features (Full)
    if not val_exists:
        print("Extracting Validation Features (Full)...")
        config.DEBUG_SAMPLE_SIZE = None  # Ensure full set
        feats, ids, labels = extract_dataset(
            config.VAL_META,
            config.TRAIN_BSON,
            model,
            mapper,
            is_test=False,
            desc="Validation Set",
        )
        np.save(config.VAL_FEATURES, feats)
        np.save(config.VAL_IDS, ids)
        np.save(config.VAL_LABELS_L3, labels)
        del feats, ids, labels

    # 3. Extract Test Features (Full)
    if not test_exists:
        print("Extracting Test Features (Full)...")
        config.DEBUG_SAMPLE_SIZE = None  # Ensure full set
        feats, ids, _ = extract_dataset(
            config.TEST_META,
            config.TEST_BSON,
            model,
            mapper,
            is_test=True,
            desc="Test Set",
        )
        np.save(config.TEST_FEATURES, feats)
        np.save(config.TEST_IDS, ids)
        del feats, ids

    print("Feature extraction complete.")


def perform_validation_and_analysis(model_paths):
    """
    Validates the ensemble on the full validation set and performs failure analysis.
    """
    print("\n=== Validation & Failure Analysis ===")
    mapper = HierarchyMapper(load_cached_data=True)

    # Load Full Validation Set
    # We explicitly use sample_size=None to load the full cached file
    val_dataset = CachedFeatureDataset(
        features_path=config.VAL_FEATURES,
        labels_path=config.VAL_LABELS_L3,
        ids_path=config.VAL_IDS,
        hierarchy_mapper=mapper,
        sample_size=None,
    )

    loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Ensemble Models
    models = []
    for path in model_paths:
        m = HierarchicalMLP().to(config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=config.DEVICE))
        m.eval()
        models.append(m)

    correct_count = 0
    total_samples = 0

    all_errors = []
    all_feature_norms = []

    print(f"Validating on {len(val_dataset)} samples...")

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(config.DEVICE)
            # targets is tuple (l1, l2, l3), we need l3
            l3_targets = targets[2].to(config.DEVICE)

            # Ensemble Inference
            avg_probs = None
            for model in models:
                # Output: (logits_l1, logits_l2, logits_l3)
                logits = model(features)[2]
                probs = torch.softmax(logits, dim=1)

                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            avg_probs /= len(models)
            _, preds = torch.max(avg_probs, dim=1)

            # Accuracy Calculation
            correct_mask = (preds == l3_targets).cpu().numpy().astype(int)
            correct_count += correct_mask.sum()
            total_samples += len(correct_mask)

            # Failure Analysis Data
            # Error = 1 if wrong, 0 if correct
            batch_errors = 1 - correct_mask
            all_errors.append(batch_errors)

            # Compute L2 norm of input features (proxy for signal strength/outlier status)
            norms = torch.norm(features, p=2, dim=1).cpu().numpy()
            all_feature_norms.append(norms)

    # Final Metric
    final_acc = correct_count / total_samples
    print(f"Final Validation Metric: {final_acc}")

    # Correlation Analysis
    all_errors = np.concatenate(all_errors)
    all_feature_norms = np.concatenate(all_feature_norms)

    corr, _ = pearsonr(all_errors, all_feature_norms)
    print(f"Correlation between Error magnitude and Input Feature L2 Norm: {corr:.6f}")

    return final_acc


def main():
    # 1. Setup
    set_seed(42)

    # 2. Configure Runtime for Fast Baseline
    # We modify config attributes directly. Since modules share the config instance,
    # these changes propagate to library modules.
    config.EPOCHS = 5
    config.ENSEMBLE_SIZE = 3

    # 3. Feature Extraction
    # We use a custom routine to handle mixed dataset sizes (Train subset vs Val full)
    custom_feature_extraction()

    # 4. Training
    # Reset sample size to None so trainer uses all data available in the cached files.
    # (Train file will have 200k, Val file will have full set)
    config.DEBUG_SAMPLE_SIZE = None
    model_paths = trainer.train_ensemble()

    # 5. Validation
    metric = perform_validation_and_analysis(model_paths)

    # 6. Submission
    threshold = 0.6239621493939094
    if metric > threshold:
        inference.generate_submission(model_paths)
    else:
        print(
            f"Validation metric {metric} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
