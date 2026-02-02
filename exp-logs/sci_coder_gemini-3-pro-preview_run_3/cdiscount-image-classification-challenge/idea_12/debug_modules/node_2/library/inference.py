import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import library.config as config
from library.dataset import CachedFeatureDataset
from library.model import HierarchicalMLP
from library.hierarchy_utils import HierarchyMapper


def generate_submission(model_paths=None, batch_size=config.BATCH_SIZE_TRAIN):
    """
    Generates predictions for the test set using the trained ensemble.

    Args:
        model_paths (list of str, optional): List of paths to trained model weights.
                                             If None, infers paths from config template.
        batch_size (int): Batch size for inference. Defaults to config.BATCH_SIZE_TRAIN.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    print("=== Generating Submission ===")

    # 1. Determine Model Paths
    if model_paths is None:
        model_paths = [
            config.MODEL_SAVE_PATH_TEMPLATE.format(i)
            for i in range(config.ENSEMBLE_SIZE)
        ]

    # Verify models exist
    valid_paths = [p for p in model_paths if os.path.exists(p)]
    if not valid_paths:
        raise FileNotFoundError(
            f"No trained models found. Expected paths like: {model_paths[0]}"
        )

    print(f"Found {len(valid_paths)} models for ensemble inference.")

    # 2. Load Hierarchy Mapper
    # Used to map predicted L3 indices back to category_ids
    mapper = HierarchyMapper(load_cached_data=True)

    # 3. Load Test Data
    # We use the CachedFeatureDataset to stream features from disk (mmap)
    # ensuring we don't overload RAM with the large test set.
    if not os.path.exists(config.TEST_FEATURES) or not os.path.exists(config.TEST_IDS):
        raise FileNotFoundError(
            f"Test features or IDs not found at {config.TEST_FEATURES}. "
            "Ensure feature extraction is complete."
        )

    test_dataset = CachedFeatureDataset(
        features_path=config.TEST_FEATURES,
        ids_path=config.TEST_IDS,
        hierarchy_mapper=mapper,
        sample_size=None,  # Always process full test set for submission
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Load Models
    models = []
    for path in valid_paths:
        m = HierarchicalMLP().to(config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=config.DEVICE))
        m.eval()
        models.append(m)

    # 5. Inference Loop
    all_preds = []
    all_ids = []

    print(f"Starting inference on {len(test_dataset)} test samples...")

    with torch.no_grad():
        for batch_idx, (features, ids) in enumerate(test_loader):
            features = features.to(config.DEVICE)

            # Ensemble Prediction Accumulator
            avg_probs = None

            for model in models:
                # Forward pass returns (logits_l1, logits_l2, logits_l3)
                # We only care about L3 (target) for the submission
                logits_l3 = model(features)[2]
                probs = torch.softmax(logits_l3, dim=1)

                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            # Average probabilities across ensemble
            avg_probs /= len(models)

            # Get predictions (Index of max probability)
            _, preds = torch.max(avg_probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_ids.append(ids.numpy())

    # 6. Post-Processing
    final_preds_idx = np.concatenate(all_preds)
    final_ids = np.concatenate(all_ids)

    # Map L3 indices back to category_id
    # mapper.l3_to_cat is an array where index i corresponds to the category_id for class i
    final_category_ids = mapper.l3_to_cat[final_preds_idx]

    # 7. Create Submission DataFrame
    submission_df = pd.DataFrame({"_id": final_ids, "category_id": final_category_ids})

    # 8. Save
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")

    return submission_df
