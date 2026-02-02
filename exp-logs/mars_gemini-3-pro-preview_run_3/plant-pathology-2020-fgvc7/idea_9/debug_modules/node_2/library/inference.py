import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data import AppleDataset, get_transforms
from library.modeling import get_model


def predict_with_tta(model, loader, device):
    """
    Performs inference on the data loader using the given model.
    Applies Test-Time Augmentation (TTA) based on Config.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Predicted probabilities of shape (N_samples, N_classes).
    """
    model.eval()
    preds = []

    # TTA Configuration
    use_hflip = Config.TTA_FLIP_HORIZONTAL
    # Note: Vertical flip and Transpose are explicitly excluded in this strategy

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            # 1. Forward pass on original images
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # 2. Forward pass on horizontally flipped images (if enabled)
            if use_hflip:
                # Flip along width dimension (dim 3: B, C, H, W)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.softmax(outputs_flipped, dim=1)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            preds.append(probs.cpu().numpy())

    return np.concatenate(preds)


def generate_submission(debug=False):
    """
    Generates the final submission file by creating an ensemble of all trained models.

    Args:
        debug (bool): If True, runs on a small subset of the test data.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Starting submission generation...")

    # Load Test Metadata
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test metadata file not found: {Config.TEST_CSV}")

    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        print(
            f"Debug mode enabled. Subsampling test set to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

    # Initialize prediction accumulator
    # Shape: (N_samples, N_classes)
    final_preds = np.zeros((len(test_df), Config.NUM_CLASSES), dtype=np.float32)
    models_used = 0

    # Define architectures to ensemble
    # These keys match the 'model_type' argument in get_model and get_transforms
    # and also the naming convention for saved weights.
    architectures = ["effnet", "maxvit"]

    for arch in architectures:
        print(f"\nProcessing architecture: {arch}")

        # Get appropriate transforms (image size differs between architectures)
        transforms = get_transforms(data="valid", model_type=arch)

        # Create Dataset and Loader
        dataset = AppleDataset(
            df=test_df, transform=transforms, output_label=False  # Test mode
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Iterate through folds
        for fold in range(Config.NUM_FOLDS):
            weight_path = os.path.join(Config.WORK_DIR, f"{arch}_fold_{fold}_best.pth")

            if not os.path.exists(weight_path):
                print(
                    f"  [Warning] Weights not found for {arch} Fold {fold} at {weight_path}. Skipping."
                )
                continue

            print(f"  - Inference with {arch} Fold {fold}...")

            # Initialize model
            # pretrained=False because we are loading custom weights
            model = get_model(model_type=arch, pretrained=False)

            # Load weights
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)

            # Predict
            fold_preds = predict_with_tta(model, loader, device)

            # Accumulate
            final_preds += fold_preds
            models_used += 1

            # Cleanup
            del model, state_dict, fold_preds
            torch.cuda.empty_cache()

    if models_used == 0:
        raise RuntimeError("No trained models were found! Cannot generate submission.")

    print(f"\nInference complete. Ensembled {models_used} models.")

    # Average predictions
    avg_preds = final_preds / models_used

    # Create Submission DataFrame
    submission_df = pd.DataFrame(avg_preds, columns=Config.CLASS_LABELS)
    submission_df.insert(0, "image_id", test_df["image_id"])

    # Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print("Submission Preview:")
    print(submission_df.head())
