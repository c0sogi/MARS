import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import BirdDataset, get_transforms
from library.model import BirdClassifier
from library.engine import validate_one_epoch


def predict_and_submit(n_folds=Config.N_FOLDS, debug=Config.DEBUG):
    """
    Loads trained models from the K-Fold cross-validation, runs inference on the test set,
    ensembles the predictions via averaging, and generates the submission file.

    Args:
        n_folds (int): Number of folds to use for the ensemble.
        debug (bool): If True, runs on a subset of data for testing purposes.
    """
    seed_everything(Config.SEED)

    # --- 1. Load Test Data ---
    if not os.path.exists(Config.TEST_CSV_PATH):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_CSV_PATH}")

    df_test = pd.read_csv(Config.TEST_CSV_PATH)

    if debug:
        print("Debug mode: utilizing subset of test data.")
        df_test = df_test.sample(n=10, random_state=Config.SEED).reset_index(drop=True)

    print(f"Starting inference on {len(df_test)} test samples...")

    # --- 2. Prepare DataLoader ---
    # We use the 'test' transforms which are deterministic (resize + normalize)
    test_dataset = BirdDataset(df_test, transforms=get_transforms(data="test"))

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # --- 3. Ensemble Inference ---
    # Array to store sum of predictions from all folds
    # Shape: (N_samples, N_classes)
    avg_preds = np.zeros((len(df_test), Config.NUM_CLASSES))
    models_used = 0

    for fold in range(n_folds):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Predicting with model fold {fold}...")

        # Initialize Model
        # We don't need to load pretrained weights from the internet (pretrained=False),
        # because we will load our specific checkpoint immediately after.
        model = BirdClassifier(
            backbone=Config.BACKBONE,
            pretrained=False,
            num_classes=Config.NUM_CLASSES,
        )
        model.to(Config.DEVICE)

        # Load weights
        try:
            load_checkpoint(model_path, model, device=Config.DEVICE)
        except Exception as e:
            print(f"Error loading checkpoint for fold {fold}: {e}")
            continue

        # Inference
        # validate_one_epoch returns (loss, predictions, targets)
        # We only care about predictions.
        # Note: The test dataset has dummy targets (0s), so loss/targets are irrelevant here.
        _, preds, _ = validate_one_epoch(model, test_loader, Config.DEVICE)

        avg_preds += preds
        models_used += 1

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    if models_used == 0:
        raise RuntimeError(
            "No models were successfully loaded. Cannot generate predictions."
        )

    # Average predictions (Bagging)
    avg_preds /= models_used
    print(f"Inference complete. Averaged predictions from {models_used} models.")

    # --- 4. Format Submission ---
    print("Formatting submission...")

    submission_rows = []
    rec_ids = df_test["rec_id"].values

    # avg_preds shape: (n_samples, n_classes)
    for i, rec_id in enumerate(rec_ids):
        probs = avg_preds[i]
        for species_idx, prob in enumerate(probs):
            # Construct Id as per competition format: rec_id * 100 + species_id
            # Example: rec_id=1, species=2 -> Id=102
            row_id = int(rec_id * 100 + species_idx)
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_submission = pd.DataFrame(submission_rows)

    # Sort by Id to ensure consistent order
    df_submission = df_submission.sort_values("Id")

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
