import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloaders
from library.models import CustomResNet, CustomDenseNet


def predict_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA).
    Predicts on: Original, Horizontal Flip, Vertical Flip.
    Returns the average probability.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The test data loader.
        device (torch.device): The computation device.

    Returns:
        np.ndarray: Array of probabilities with shape (N,).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original
            outputs_orig = model(images)
            probs_orig = torch.sigmoid(outputs_orig)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            outputs_h = model(images_h)
            probs_h = torch.sigmoid(outputs_h)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            outputs_v = model(images_v)
            probs_v = torch.sigmoid(outputs_v)

            # Average predictions
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0
            all_preds.append(avg_probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def generate_ensemble_predictions():
    """
    Loads all trained models (architectures x seeds), performs TTA inference,
    aggregates predictions via averaging, and saves the submission file.
    """
    # Setup
    Config.setup_directories()
    device = Config.DEVICE

    print("Initializing Inference...")

    # Load Test Metadata to get IDs in correct order
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    test_ids = df_test_meta["id"].values

    # Get DataLoader
    # Note: We use shuffle=False for test_loader in get_dataloaders to preserve order
    _, _, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # Accumulator for ensemble predictions
    ensemble_preds = np.zeros(len(test_ids))
    model_count = 0

    # Iterate over all defined architectures and seeds
    for arch in Config.ARCHITECTURES:
        for seed in Config.SEEDS:
            model_path = Config.get_model_path(arch, seed)

            if not os.path.exists(model_path):
                print(f"Warning: Model checkpoint not found at {model_path}. Skipping.")
                continue

            print(f"Predicting with {arch} (seed {seed})...")

            # Instantiate Model
            if arch == "resnet":
                model = CustomResNet(num_classes=Config.NUM_CLASSES)
            elif arch == "densenet":
                model = CustomDenseNet(num_classes=Config.NUM_CLASSES)
            else:
                raise ValueError(f"Unknown architecture: {arch}")

            # Load Weights
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)

            # Predict with TTA
            preds = predict_tta(model, test_loader, device)

            # Verify shape alignment
            if len(preds) != len(ensemble_preds):
                # If debug mode was used during training/inference, shapes might mismatch
                # We assume standard execution flow here.
                print(
                    f"Shape mismatch: Preds {len(preds)} vs Total {len(ensemble_preds)}"
                )
                # Adjust accumulator size if necessary (e.g. if debug flag changed)
                if Config.DEBUG:
                    ensemble_preds = ensemble_preds[: len(preds)]
                    test_ids = test_ids[: len(preds)]

            ensemble_preds += preds
            model_count += 1

    if model_count == 0:
        raise RuntimeError("No models were loaded. Cannot generate predictions.")

    # Average predictions
    final_preds = ensemble_preds / model_count

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
