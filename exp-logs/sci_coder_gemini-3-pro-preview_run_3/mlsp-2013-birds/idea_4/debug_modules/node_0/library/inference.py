import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders
from library.model import BirdResNet


def predict_ensemble(load_cached_data=True):
    """
    Runs inference on the test set using an ensemble of trained models from all folds.
    Generates a submission file in the format required by the competition.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed spectrograms
                                 from the cache directory.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"{'='*20} Starting Ensemble Inference {'='*20}")

    # 1. Load Test Metadata
    if not os.path.exists(Config.TEST_CSV):
        print(f"Error: Test metadata not found at {Config.TEST_CSV}")
        return

    test_df = pd.read_csv(Config.TEST_CSV)
    print(f"Loaded test metadata: {len(test_df)} samples")

    # 2. Prepare DataLoader
    # We pass empty DataFrames for train/val as we only need the test loader
    _, _, test_loader = get_dataloaders(
        pd.DataFrame(), pd.DataFrame(), test_df, load_cached_data=load_cached_data
    )

    # 3. Initialize Ensemble Variables
    device = Config.DEVICE
    num_classes = Config.NUM_CLASSES
    num_folds = Config.NUM_FOLDS

    # Shape: (N_samples, N_classes)
    ensemble_preds = np.zeros((len(test_df), num_classes), dtype=np.float32)
    models_found = 0

    # 4. Iterate Over Folds
    for fold_idx in range(num_folds):
        checkpoint_filename = f"fold_{fold_idx}_best.pth"
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint for fold {fold_idx} not found at {checkpoint_path}. Skipping."
            )
            continue

        print(f"Loading model for Fold {fold_idx}...")

        # Initialize model architecture
        # pretrained=False because we are loading custom weights immediately
        model = BirdResNet(pretrained=False, num_classes=num_classes)
        model.to(device)

        # Load weights
        load_checkpoint(checkpoint_filename, model, device=device)
        model.eval()

        # Inference Loop for this fold
        fold_preds = []
        with torch.no_grad():
            for images, _, _ in test_loader:
                images = images.to(device)

                # Forward pass
                logits = model(images)

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                fold_preds.append(probs.cpu().numpy())

        # Concatenate batches
        if fold_preds:
            fold_preds = np.concatenate(fold_preds, axis=0)
            ensemble_preds += fold_preds
            models_found += 1
        else:
            print(f"Warning: No predictions generated for fold {fold_idx}.")

    # 5. Average Predictions
    if models_found > 0:
        print(f"Averaging predictions from {models_found} models.")
        ensemble_preds /= models_found
    else:
        print("Error: No valid models found. Cannot generate submission.")
        return

    # 6. Format Submission
    # Format: Id,Probability
    # Id = rec_id * 100 + species_id
    print("Formatting submission...")

    submission_rows = []
    rec_ids = test_df["rec_id"].values

    # Ensure alignment: test_loader preserves order of test_df
    if len(ensemble_preds) != len(rec_ids):
        print(
            f"Error: Mismatch between predictions ({len(ensemble_preds)}) and recording IDs ({len(rec_ids)})"
        )
        return

    for i, rec_id in enumerate(rec_ids):
        probs = ensemble_preds[i]
        for species_id in range(num_classes):
            row_id = int(rec_id * 100 + species_id)
            prob = probs[species_id]
            submission_rows.append({"Id": row_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # 7. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"{'='*20} Inference Complete {'='*20}")
