import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import CFG
from library.utils import seed_everything
from library.data import AppleDataset, get_transforms, load_test_data
from library.modeling import get_model


def predict_all_folds(debug: bool = False):
    """
    Performs inference on the test set using the Heterogeneous K-Fold Ensemble.

    Applies Test-Time Augmentation (TTA) by averaging predictions of the original
    and horizontally flipped images. Aggregates predictions across all available
    folds and architectures.

    Args:
        debug (bool): If True, runs on a subset of data for debugging.
    """
    seed_everything(CFG.seed)
    device = torch.device(CFG.device)

    print(f"Starting inference on device: {device}")

    # 1. Load Test Data
    try:
        test_df = load_test_data(debug=debug)
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        return

    # Use 'test' transforms which only resize and normalize
    test_dataset = AppleDataset(test_df, transform=get_transforms("test"))

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Initialize array to store sum of probabilities from all models
    # Shape: (n_test_samples, n_classes)
    avg_preds = np.zeros((len(test_df), CFG.num_classes), dtype=np.float32)

    models_found = 0

    # 2. Iterate through Architectures and Folds
    for arch in CFG.model_architectures:
        for fold in range(CFG.n_folds):
            model_path = os.path.join(CFG.models_dir, f"{arch}_fold_{fold}.pth")

            if not os.path.exists(model_path):
                print(f"Checkpoint not found: {model_path}. Skipping.")
                continue

            print(f"Processing {arch} | Fold {fold}...")

            # Initialize model
            # pretrained=False because we are loading custom weights
            model = get_model(arch, CFG.num_classes, pretrained=False)

            # Load weights
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Failed to load weights for {model_path}: {e}")
                continue

            model.to(device)
            model.eval()

            fold_preds = []

            # 3. Inference Loop
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)

                    # TTA Step 1: Original Image
                    output_orig = model(images)
                    probs_orig = torch.softmax(output_orig, dim=1)

                    # TTA Step 2: Horizontal Flip
                    # Flip along width dimension (dim=3 for NCHW)
                    images_flipped = torch.flip(images, dims=[3])
                    output_flip = model(images_flipped)
                    probs_flip = torch.softmax(output_flip, dim=1)

                    # Average TTA predictions
                    batch_preds = (probs_orig + probs_flip) / 2.0

                    fold_preds.append(batch_preds.cpu().numpy())

            # Concatenate batch predictions for this model
            fold_preds = np.concatenate(fold_preds, axis=0)

            # Add to ensemble accumulator
            avg_preds += fold_preds
            models_found += 1

            # Cleanup to save memory
            del model
            torch.cuda.empty_cache()

    # 4. Finalize Predictions
    if models_found == 0:
        print("Error: No models were found/loaded. Cannot generate submission.")
        return

    print(f"Aggregating predictions from {models_found} models.")
    avg_preds /= models_found

    # 5. Create Submission DataFrame
    submission = pd.DataFrame(avg_preds, columns=CFG.target_cols)

    # Insert image_id at the beginning
    submission.insert(0, "image_id", test_df["image_id"])

    # Save to disk
    os.makedirs(os.path.dirname(CFG.submission_path), exist_ok=True)
    submission.to_csv(CFG.submission_path, index=False)

    print(f"Submission saved to {CFG.submission_path}")
    print("First 5 rows of submission:")
    print(submission.head())
