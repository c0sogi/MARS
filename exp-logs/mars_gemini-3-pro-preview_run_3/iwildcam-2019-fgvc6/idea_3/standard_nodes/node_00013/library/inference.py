import os
import torch
import pandas as pd
from torch.cuda.amp import autocast
from library.config import Config
from library.utils import get_logger
from library.dataset import get_dataloaders
from library.model import AnimalEfficientNet

# Initialize logger
logger = get_logger("inference")


def predict_tta(model, test_loader, device):
    """
    Performs inference with Test-Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        list: A list of dictionaries containing 'Id' and 'Predicted' class.
    """
    model.eval()
    results = []

    logger.info(f"Starting TTA Inference (Flip={Config.TTA_FLIP})...")

    with torch.no_grad():
        for batch_idx, (images, ids) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)

            # Mixed Precision Inference
            with autocast():
                # 1. Forward pass with original images
                logits_orig = model(images)

                if Config.TTA_FLIP:
                    # 2. Forward pass with horizontally flipped images
                    # dim 3 is width for (B, C, H, W)
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flip = model(images_flipped)

                    # 3. Average logits
                    logits = (logits_orig + logits_flip) / 2.0
                else:
                    logits = logits_orig

            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            # Store results
            for img_id, pred in zip(ids, preds):
                results.append({"Id": img_id, "Category": pred})

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    return results


def generate_submission(checkpoint_path, debug=Config.DEBUG):
    """
    Generates the submission file using the best trained model.

    Args:
        checkpoint_path (str): Path to the model checkpoint (.pth file).
        debug (bool): Whether to run in debug mode (subset of data).
    """
    device = Config.DEVICE

    # 1. Load Data
    logger.info("Loading test data...")
    # We only need the test_loader, ignore train/val
    _, _, test_loader = get_dataloaders(debug=debug, load_cached_data=True)

    # 2. Initialize Model
    logger.info(f"Initializing model: {Config.MODEL_NAME}")
    model = AnimalEfficientNet(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,  # No need to download pretrained weights, we load checkpoint
        drop_path_rate=0.0,  # No stochastic depth during inference
        use_gem=Config.USE_GEM_POOLING,
    )
    model.to(device)

    # 3. Load Weights
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # 4. Run Inference
    predictions = predict_tta(model, test_loader, device)

    # 5. Create Submission DataFrame
    submission_df = pd.DataFrame(predictions)

    # 6. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Submission shape: {submission_df.shape}")
    logger.info(f"Head:\n{submission_df.head()}")
