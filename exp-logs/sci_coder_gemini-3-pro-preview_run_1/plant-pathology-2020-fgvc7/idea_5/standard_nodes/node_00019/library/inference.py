import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import AppleLeafDataset, get_transforms
from library.models import get_model
from library.utils import seed_everything


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    Strategy: Average predictions of the original image and a horizontally flipped version.

    Args:
        model (torch.nn.Module): The trained model in evaluation mode.
        dataloader (DataLoader): DataLoader for the test set.
        device (torch.device): The device to run inference on.

    Returns:
        tuple: (predictions numpy array of shape [N, num_classes], list of image_ids)
    """
    model.eval()
    all_probs = []
    image_ids = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            # 1. Prediction on original images
            outputs_orig = model(images)
            probs_orig = torch.softmax(outputs_orig, dim=1)

            # 2. Prediction on horizontally flipped images
            # Images are (B, C, H, W). Flip on W (dim 3).
            images_flipped = torch.flip(images, dims=[3])
            outputs_flipped = model(images_flipped)
            probs_flipped = torch.softmax(outputs_flipped, dim=1)

            # 3. Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            all_probs.append(avg_probs.cpu().numpy())
            image_ids.extend(ids)

    return np.concatenate(all_probs, axis=0), image_ids


def generate_submission(load_cached_data=False):
    """
    Generates the submission file by ensembling predictions from all trained models.

    This function:
    1. Loads the test metadata.
    2. Iterates through all defined architectures and folds.
    3. Loads each model and performs inference with TTA.
    4. Averages the probabilities across all models.
    5. Saves the final predictions to a CSV file.

    Args:
        load_cached_data (bool): Flag to indicate if cached data should be used.
                                 (Not primarily used here as inference is always run fresh
                                 to ensure it matches the latest models, but kept for signature consistency).
    """
    seed_everything(Config.SEED)

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 1. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    print(f"Loaded test metadata: {len(test_df)} samples.")

    # 2. Prepare DataLoader
    # We use shuffle=False to maintain order, though we also track IDs explicitly.
    test_dataset = AppleLeafDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Accumulators
    num_samples = len(test_df)
    num_classes = Config.NUM_CLASSES
    aggregated_preds = np.zeros((num_samples, num_classes), dtype=np.float32)
    model_count = 0
    final_image_ids = None

    models_dir = os.path.join(Config.WORKING_DIR, "models")

    print("Starting ensemble inference...")

    # 4. Iterate over Architectures and Folds
    for arch in Config.MODEL_ARCHS:
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(models_dir, f"{arch}_fold_{fold}.pth")

            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} not found. Skipping.")
                continue

            print(f"Predicting with {arch} (Fold {fold})...")

            # Load Model
            # pretrained=False speeds up loading since we overwrite weights anyway
            model = get_model(arch, num_classes, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            model.to(Config.DEVICE)

            # Inference
            preds, ids = predict_with_tta(model, test_loader, Config.DEVICE)

            # Consistency Check
            if final_image_ids is None:
                final_image_ids = ids
            elif final_image_ids != ids:
                raise ValueError(f"Image ID mismatch in {arch} fold {fold}.")

            # Accumulate
            aggregated_preds += preds
            model_count += 1

            # Cleanup
            del model
            torch.cuda.empty_cache()

    if model_count == 0:
        raise RuntimeError("No models found. Cannot generate submission.")

    # 5. Average and Save
    print(f"Averaging predictions from {model_count} models...")
    final_preds = aggregated_preds / model_count

    submission_df = pd.DataFrame(final_preds, columns=Config.CLASS_LABELS)
    submission_df.insert(0, "image_id", final_image_ids)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
    print("First 5 rows of submission:")
    print(submission_df.head())
