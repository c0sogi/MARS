import os
import torch
import numpy as np
import pandas as pd

from library.config import Config
from library.model import LSEIsomorphicCNN
from library.data_loader import get_test_loader
from library.utils import set_seed, load_checkpoint


def generate_submission():
    """
    Loads the trained models from the 5-fold cross-validation, performs inference
    on the test set (ensemble averaging), and generates the submission CSV file.
    """
    # 1. Setup Environment
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting Inference on device: {device}")

    # 2. Load Test Data
    # load_cached_data=True allows using pre-processed .npy files if they exist
    test_loader = get_test_loader(load_cached_data=True)
    print(f"Test loader ready. Batches: {len(test_loader)}")

    # 3. Load Models (Ensemble)
    models = []
    for fold_idx in range(Config.N_FOLDS):
        # Instantiate fresh model
        model = LSEIsomorphicCNN().to(device)

        # Construct checkpoint path
        # Matches the saving logic in train.py: "model_best_fold_{fold_idx}.pth"
        checkpoint_filename = f"model_best_fold_{fold_idx}.pth"
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint not found for fold {fold_idx} at {checkpoint_path}. Skipping."
            )
            continue

        # Load weights
        load_checkpoint(checkpoint_path, model, device=Config.DEVICE)

        # Set to eval mode
        model.eval()
        models.append(model)
        print(f"Loaded model for fold {fold_idx}")

    if not models:
        raise RuntimeError("No models were loaded. Cannot proceed with inference.")

    # 4. Inference Loop
    all_ids = []
    all_probs = []

    print("Running inference...")

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, angles, ids in test_loader:
            # Move inputs to device
            images = images.to(device)
            angles = angles.to(device)

            # Accumulator for ensemble probabilities
            batch_ensemble_probs = torch.zeros(images.size(0), 1, device=device)

            for model in models:
                # Forward pass
                logits = model(images, angles)
                # Sigmoid to get probabilities [0, 1]
                probs = torch.sigmoid(logits)
                batch_ensemble_probs += probs

            # Average across folds
            batch_ensemble_probs /= len(models)

            # Store results
            # Flatten to 1D array
            batch_probs_np = batch_ensemble_probs.cpu().numpy().flatten()

            all_ids.extend(ids)
            all_probs.extend(batch_probs_np)

    # 5. Generate Submission File
    submission_df = pd.DataFrame({"id": all_ids, "is_iceberg": all_probs})

    # Ensure output directory exists (handled by Config.setup, but double check)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    # Float format %.4f is usually sufficient, but we keep full precision as default or specific if needed
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(submission_df)}")
    print("Inference complete.")
