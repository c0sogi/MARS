import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from library.configuration import Config
from library.utilities import load_checkpoint, save_submission
from library.data_loader import get_data_arrays, IcebergDataset
from library.architecture import IcebergResNet


def get_test_loader(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, limit=None
):
    """
    Creates the test dataloader.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        limit (int, optional): Limit number of samples for debugging.
    """
    # Load data arrays (cached)
    # Returns: train_images, train_angles, train_labels, test_images, test_angles, test_ids
    _, _, _, test_imgs, test_angles, test_ids = get_data_arrays(load_cached_data=True)

    # Optional debugging limit
    if limit is not None:
        test_imgs = test_imgs[:limit]
        test_angles = test_angles[:limit]
        test_ids = test_ids[:limit]

    # Transform (just ToTensor, normalization done in preprocessing)
    transform = A.Compose([ToTensorV2()])

    dataset = IcebergDataset(
        test_imgs, test_angles, ids=test_ids, transform=transform, mode="test"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


def load_ensemble_models(device, num_models=Config.NUM_ENSEMBLE_MODELS):
    """
    Loads the trained SWA ensemble models.

    Args:
        device (str): Device to load models onto.
        num_models (int): Number of models in the ensemble.

    Returns:
        list: List of loaded PyTorch models.
    """
    models = []
    print(f"Loading {num_models} ensemble models from {Config.CHECKPOINT_DIR}...")

    for i in range(num_models):
        model = IcebergResNet().to(device)
        filename = f"ensemble_{i}_swa.pth"

        try:
            # load_checkpoint handles the 'module.' prefix stripping from SWA/DataParallel
            load_checkpoint(model, filename=filename)
            model.eval()
            models.append(model)
            print(f"Loaded model {i + 1}/{num_models}: {filename}")
        except FileNotFoundError:
            print(f"Warning: Model checkpoint {filename} not found. Skipping.")

    if not models:
        raise RuntimeError("No models were loaded. Check checkpoint directory.")

    return models


def predict_with_tta(models, loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA) and Model Ensembling.

    TTA Views:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip

    Args:
        models (list): List of loaded models.
        loader (DataLoader): Test data loader.
        device (str): Device to run inference on.

    Returns:
        tuple: (list of ids, list of probabilities)
    """
    all_probs = []
    all_ids = []

    print("Starting inference with TTA...")

    with torch.no_grad():
        for i, (images, angles, ids) in enumerate(loader):
            images = images.to(device)
            angles = angles.to(device)

            # Define TTA Views
            # 1. Original
            # 2. Horizontal Flip (dim 3 for N,C,H,W)
            # 3. Vertical Flip (dim 2 for N,C,H,W)
            views = [images, torch.flip(images, dims=[3]), torch.flip(images, dims=[2])]

            # Accumulate probabilities
            batch_probs_sum = torch.zeros(images.size(0), device=device)

            for model in models:
                for view in views:
                    # Forward pass
                    logits = model(view, angles)
                    # Convert to probability
                    probs = torch.sigmoid(logits).view(-1)
                    batch_probs_sum += probs

            # Average over (Models * Views)
            num_predictions = len(models) * len(views)
            avg_probs = batch_probs_sum / num_predictions

            all_probs.extend(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    return all_ids, all_probs


def run_inference_pipeline(limit=None):
    """
    Main function to execute the inference pipeline.

    Args:
        limit (int, optional): Limit dataset size for debugging.
    """
    device = Config.DEVICE

    # 1. Prepare Data
    loader = get_test_loader(limit=limit)

    # 2. Load Models
    models = load_ensemble_models(device)

    # 3. Predict
    ids, probs = predict_with_tta(models, loader, device)

    # 4. Save
    save_submission(ids, probs, "submission.csv")
    print(
        f"Inference completed. Submission saved to {Config.SUBMISSION_DIR}/submission.csv"
    )
