import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.utils import get_device
from library.model import MicroResNet
from library.dataset import IcebergDataset, CACHE_DIR
from library.trainer import load_test_data


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).

    Strategies:
    1. Original Image
    2. Horizontal Flip
    3. Vertical Flip

    Args:
        model (nn.Module): The trained PyTorch model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)

            # 1. Original
            pred_orig = model(images, angles)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, dims=[3])
            pred_h = model(images_h, angles)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, dims=[2])
            pred_v = model(images_v, angles)

            # Average predictions
            pred_avg = (pred_orig + pred_h + pred_v) / 3.0

            preds_list.extend(pred_avg.cpu().numpy())

    return np.array(preds_list)


def generate_submission(
    n_splits=5,
    batch_size=32,
    num_workers=2,
    load_cached_data=True,
    angle_mean=0.0,
    output_path="./submission/submission.csv",
):
    """
    Generates the final submission file by ensembling predictions from multiple folds
    using Test-Time Augmentation.

    Args:
        n_splits (int): Number of cross-validation folds (models) to load.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker threads for DataLoader.
        load_cached_data (bool): Whether to use cached preprocessed data.
        angle_mean (float): Mean incidence angle to use for imputation if needed.
        output_path (str): Path to save the submission CSV.
    """
    device = get_device()

    # 1. Load Test Data
    # We use the utility from trainer.py to ensure consistent processing/caching
    X_test, ang_test, ids_test = load_test_data(
        load_cached_data=load_cached_data, angle_mean=angle_mean
    )

    # 2. Create Dataset and Loader
    # No transform needed here as TTA is handled manually in the prediction loop
    test_dataset = IcebergDataset(
        X_test, ang_test, y=None, ids=ids_test, transform=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Ensemble Prediction
    ensemble_preds = np.zeros(len(ids_test))
    models_found = 0

    print(f"Starting inference on {len(ids_test)} samples using {n_splits} folds...")

    for fold in range(n_splits):
        fold_dir = os.path.join(CACHE_DIR, f"fold_{fold}")
        model_path = os.path.join(fold_dir, "model_best.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Predicting with Fold {fold} model...")

        # Initialize and Load Model
        model = MicroResNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

        # Predict with TTA
        fold_preds = predict_with_tta(model, test_loader, device)

        ensemble_preds += fold_preds
        models_found += 1

    if models_found == 0:
        raise RuntimeError(
            "No trained models found in the working directory. Cannot generate submission."
        )

    # Average over the number of models found
    ensemble_preds /= models_found

    # 4. Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": ensemble_preds})

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
