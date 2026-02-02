import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import FeatureDataset
from library.model import HierarchicalMLP
from library.utils import HierarchyMap


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def predict_ensemble(batch_size=Config.TRAIN_BATCH_SIZE, device=None):
    """
    Performs inference using the trained ensemble and generates submission.csv.

    Args:
        batch_size (int): Batch size for inference. Defaults to Config.TRAIN_BATCH_SIZE.
        device (torch.device, optional): Device to run inference on. If None, detects automatically.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    set_seed(Config.SEED)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting Ensemble Inference on {device}...")

    # Load Hierarchy Map for decoding labels later
    # We must use cached data to ensure the encoding matches the training phase
    hierarchy_map = HierarchyMap(load_cached_data=True)

    # Prepare Test Dataset
    # We assume feature extraction has already been run and saved to Config.TEST_FEATURES
    if not os.path.exists(Config.TEST_FEATURES) or not os.path.exists(Config.TEST_IDS):
        raise FileNotFoundError(
            f"Test features or IDs not found at {Config.TEST_FEATURES} / {Config.TEST_IDS}. "
            "Please run feature extraction first."
        )

    test_dataset = FeatureDataset(
        feature_path=Config.TEST_FEATURES, id_path=Config.TEST_IDS, mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load Models
    models = []
    # Check for models in the directory defined in Config
    if not os.path.exists(Config.MODEL_DIR):
        raise FileNotFoundError(f"Model directory {Config.MODEL_DIR} does not exist.")

    for i in range(Config.ENSEMBLE_SIZE):
        model_path = os.path.join(Config.MODEL_DIR, f"ensemble_model_{i}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found. Skipping.")
            continue

        print(f"Loading model: {model_path}")
        model = HierarchicalMLP().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError("No trained models found for inference.")

    all_preds = []
    all_ids = []

    print(
        f"Running inference on {len(test_dataset)} samples with {len(models)} models..."
    )

    with torch.no_grad():
        for features, product_ids in test_loader:
            features = features.to(device)

            # Aggregate probabilities from all models
            # Shape: (Batch_Size, Num_Classes_L3)
            ensemble_probs = torch.zeros(
                features.size(0), Config.NUM_CLASSES_L3, device=device
            )

            for model in models:
                # Forward pass returns (l1_logits, l2_logits, l3_logits)
                _, _, l3_logits = model(features)

                # Apply Softmax to get probabilities
                probs = torch.softmax(l3_logits, dim=1)
                ensemble_probs += probs

            # Average probabilities across the ensemble
            ensemble_probs /= len(models)

            # Get final class predictions (index with highest probability)
            _, preds = torch.max(ensemble_probs, 1)

            all_preds.append(preds.cpu().numpy())
            all_ids.append(product_ids.numpy())

    # Concatenate results from all batches
    final_preds = np.concatenate(all_preds)
    final_ids = np.concatenate(all_ids)

    # Decode labels
    # Convert integer indices back to original category_id using the inverse transform
    print("Decoding predictions...")
    final_category_ids = hierarchy_map.l3_encoder.inverse_transform(final_preds)

    # Create DataFrame matching the submission format
    submission_df = pd.DataFrame({"_id": final_ids, "category_id": final_category_ids})

    # Save submission file
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return submission_df
