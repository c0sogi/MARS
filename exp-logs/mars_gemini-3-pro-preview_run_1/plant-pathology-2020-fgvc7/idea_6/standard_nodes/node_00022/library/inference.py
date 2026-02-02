import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger
from library.dataset import AppleDataset, get_transforms
from library.model import AppleResNet

logger = get_logger("inference")


def predict(debug: bool = False):
    """
    Generates predictions for the test set using trained models and Test-Time Augmentation (TTA).
    Saves the result to the submission file defined in Config.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data for debugging purposes.
    """
    logger.info("Starting Inference...")

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 1. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        logger.error(f"Test metadata not found at {Config.TEST_METADATA_PATH}")
        return

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        logger.info(f"Debug mode: Inference on {len(test_df)} samples.")

    # 2. Prepare Dataset and DataLoader
    # We use the 'test' transforms which only include resizing and normalization
    test_dataset = AppleDataset(
        df=test_df, transform=get_transforms("test"), data_root=Config.INPUT_DIR
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Trained Models
    # We expect 5 models from the 5-fold cross-validation
    models = []
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"resnet34_fold_{fold}.pth")

        if os.path.exists(model_path):
            try:
                model = AppleResNet()
                # Load weights
                state_dict = torch.load(model_path, map_location=Config.DEVICE)
                model.load_state_dict(state_dict)

                model.to(Config.DEVICE)
                model.eval()
                models.append(model)
                logger.info(f"Successfully loaded model for Fold {fold}")
            except Exception as e:
                logger.error(f"Failed to load model for Fold {fold}: {e}")
        else:
            logger.warning(f"Model file not found for Fold {fold} at {model_path}")

    if not models:
        logger.error("No trained models found. Aborting inference.")
        return

    # 4. Inference Loop with TTA
    all_preds = []
    image_ids = []

    logger.info(
        f"Running inference with {len(models)} models and TTA (Original + HFlip + VFlip)..."
    )

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            images = data["image"].to(Config.DEVICE)
            ids = data["image_id"]
            image_ids.extend(ids)

            # Prepare TTA versions
            # 1. Original
            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, dims=[3])
            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, dims=[2])

            batch_ensemble_preds = []

            for model in models:
                # Get logits for each version
                logits_orig = model(images)
                logits_h = model(images_h)
                logits_v = model(images_v)

                # Convert to probabilities
                probs_orig = torch.softmax(logits_orig, dim=1)
                probs_h = torch.softmax(logits_h, dim=1)
                probs_v = torch.softmax(logits_v, dim=1)

                # Average TTA for this specific model
                probs_avg_model = (probs_orig + probs_h + probs_v) / 3.0
                batch_ensemble_preds.append(probs_avg_model.cpu().numpy())

            # Average across all models (Ensemble)
            # Shape of batch_ensemble_preds: (N_Models, Batch_Size, N_Classes)
            # Mean over axis 0 -> (Batch_Size, N_Classes)
            final_batch_preds = np.mean(batch_ensemble_preds, axis=0)
            all_preds.append(final_batch_preds)

    # 5. Aggregate and Save
    all_preds = np.concatenate(all_preds, axis=0)

    # Create DataFrame
    # Columns must strictly follow the order in Config.CLASS_LABELS
    submission = pd.DataFrame(all_preds, columns=Config.CLASS_LABELS)
    submission.insert(0, "image_id", image_ids)

    # Save to CSV
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info("First 5 rows of submission:")
    # Printing full precision as requested implicitly by showing raw values
    print(submission.head().to_string())
