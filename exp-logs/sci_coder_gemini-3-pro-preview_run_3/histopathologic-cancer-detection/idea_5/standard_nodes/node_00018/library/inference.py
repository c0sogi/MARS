import os
import numpy as np
import joblib
import torch
import pandas as pd

from library.config import Config
from library.utils import seed_everything
from library.dataset import load_metadata, get_dataloaders
from library.modeling import get_model, predict_with_tta
from library.meta_learner import generate_submission


def run_inference(load_cached_data=True):
    """
    Executes the inference pipeline:
    1. Loads test data.
    2. Generates predictions for each base model (averaging across folds).
       - Uses caching for aggregated predictions per architecture.
    3. Loads the trained Meta-Learner.
    4. Generates the final submission file.

    Args:
        load_cached_data (bool): Whether to load aggregated test predictions from cache.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print("--- Starting Inference ---")

    # 2. Load Test Data
    print("Loading test metadata...")
    # load_metadata returns train, val, test. We only need test.
    _, _, test_df = load_metadata()

    if test_df is None:
        raise ValueError("Test metadata not found.")

    print(f"Test samples: {len(test_df)}")

    # Extract IDs to ensure alignment
    test_ids = test_df["id"].values.tolist()

    # Get DataLoader
    loaders = get_dataloaders(test_df=test_df, batch_size=Config.BATCH_SIZE)
    test_loader = loaders["test"]

    # 3. Load Meta-Learner
    meta_learner_path = os.path.join(Config.WORKING_DIR, "meta_learner.joblib")
    if not os.path.exists(meta_learner_path):
        raise FileNotFoundError(
            f"Meta-learner not found at {meta_learner_path}. Please train it first."
        )

    print(f"Loading meta-learner from {meta_learner_path}")
    meta_learner = joblib.load(meta_learner_path)

    # 4. Generate Base Model Predictions
    base_test_preds = {}

    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    for model_name in Config.MODELS:
        print(f"\nProcessing architecture: {model_name}")

        cache_path = os.path.join(Config.WORKING_DIR, f"test_preds_{model_name}.npy")
        avg_preds = None

        # Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached predictions from {cache_path}")
            try:
                avg_preds = np.load(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Regenerating...")
                avg_preds = None

        # Compute if not cached
        if avg_preds is None:
            fold_preds_list = []

            # Iterate over all folds
            for fold_idx in range(Config.N_FOLDS):
                model_path = os.path.join(
                    Config.WORKING_DIR, f"{model_name}_fold_{fold_idx}.pth"
                )

                if not os.path.exists(model_path):
                    print(
                        f"Warning: Model checkpoint not found: {model_path}. Skipping fold."
                    )
                    continue

                print(f"  Predicting with Fold {fold_idx}...")

                # Load Model
                # pretrained=False because we are loading custom weights
                model = get_model(model_name, pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)

                # Predict with TTA
                # predict_with_tta returns (preds_array, ids_list)
                preds, _ = predict_with_tta(model, test_loader, device)

                fold_preds_list.append(preds)

            if not fold_preds_list:
                raise RuntimeError(f"No valid folds found for model {model_name}")

            # Average predictions across folds
            avg_preds = np.mean(fold_preds_list, axis=0)

            # Save to cache
            np.save(cache_path, avg_preds)
            print(f"Saved aggregated predictions to {cache_path}")

        base_test_preds[model_name] = avg_preds

    # 5. Generate Submission
    # This function stacks the base predictions and uses the meta-learner to predict final probabilities
    generate_submission(meta_learner, base_test_preds, test_ids)
    print("Inference complete.")
