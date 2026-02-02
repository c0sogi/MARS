import os
import numpy as np
import pandas as pd
import torch
from library.utils import get_device
from library.model import IcebergCNN, predict


def generate_submission(
    test_loader,
    checkpoint_dir="./checkpoints",
    output_path="./submission/submission.csv",
    num_folds=5,
):
    """
    Generates predictions using an ensemble of trained models and saves to CSV.

    This function loads model checkpoints for each fold, performs inference on the
    test set, averages the probabilities (soft voting), and writes the result to a file.

    Args:
        test_loader (DataLoader): DataLoader for the test set.
        checkpoint_dir (str): Directory containing model checkpoints.
        output_path (str): Path to save the submission CSV.
        num_folds (int): Number of folds/models to ensemble.
    """
    device = get_device()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating submission using {num_folds} folds on {device}...")

    # Initialize array to store sum of predictions
    total_preds = None
    models_found = 0

    for fold_idx in range(num_folds):
        checkpoint_path = os.path.join(
            checkpoint_dir, f"model_best_fold_{fold_idx}.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}. Skipping fold {fold_idx}.")
            continue

        print(f"Processing Fold {fold_idx}...")

        # Initialize model architecture
        model = IcebergCNN().to(device)

        # Load trained weights
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            continue

        # Generate predictions for this fold
        # The predict function in library.model handles evaluation mode and sigmoid activation
        fold_preds = predict(model, test_loader, device)

        if total_preds is None:
            total_preds = fold_preds
        else:
            total_preds += fold_preds

        models_found += 1

        # Clean up to save memory
        del model
        torch.cuda.empty_cache()

    if models_found == 0:
        raise FileNotFoundError(f"No valid model checkpoints found in {checkpoint_dir}")

    # Average the predictions across all found models
    avg_preds = total_preds / models_found

    # Retrieve IDs from the dataset
    ids = test_loader.dataset.ids

    # Create DataFrame conforming to submission format
    submission_df = pd.DataFrame({"id": ids, "is_iceberg": avg_preds})

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} (Ensembled {models_found} models)")
