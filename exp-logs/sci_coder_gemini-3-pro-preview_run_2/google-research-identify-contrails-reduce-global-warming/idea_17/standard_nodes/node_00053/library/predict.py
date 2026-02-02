import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, get_logger, rle_encode
from library.dataset import ContrailDataset, get_transforms
from library.model import SingleStreamUNet


def predict_and_submit(load_cached_data=False):
    """
    Performs inference on the test set using the trained model and generates a submission file.

    Args:
        load_cached_data (bool): Flag to satisfy the interface requirement.
                                 Since inference generates the final output directly,
                                 this implementation proceeds to generate the submission
                                 freshly to ensure correctness.
    """
    # 1. Setup
    set_seed(Config.SEED)
    logger = get_logger("predict")
    device = torch.device(Config.DEVICE)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    logger.info(f"Starting inference on device: {device}")

    # 2. Data Loading
    # We use the test metadata path defined in Config
    test_dataset = ContrailDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        split="test",
        transform=get_transforms("test"),
        debug=Config.DEBUG,
    )

    # If dataset is empty (e.g. dummy metadata), handle gracefully
    if len(test_dataset) == 0:
        logger.warning("Test dataset is empty. Creating empty submission.")
        df_sub = pd.DataFrame(columns=["record_id", "encoded_pixels"])
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Loading
    model = SingleStreamUNet(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # No need to download pretrained weights for inference, we load our own
        in_chans=Config.IN_CHANNELS,
    )

    # Load trained weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        logger.info(f"Loading model weights from {Config.BEST_MODEL_PATH}")
        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        logger.error(f"Model weights not found at {Config.BEST_MODEL_PATH}")
        raise FileNotFoundError(f"Model weights not found at {Config.BEST_MODEL_PATH}")

    model.to(device)
    model.eval()

    # 4. Inference Loop
    results = []
    threshold = Config.THRESHOLD

    logger.info(f"Running inference with TTA={Config.USE_TTA}...")

    with torch.no_grad():
        for batch_idx, (images, _, record_ids) in enumerate(test_loader):
            images = images.to(device, dtype=torch.float32)

            # --- Test-Time Augmentation (TTA) ---
            if Config.USE_TTA:
                # 1. Original
                logits_1 = model(images)
                probs_1 = torch.sigmoid(logits_1)

                # 2. Horizontal Flip (dim 3)
                images_h = torch.flip(images, [3])
                logits_2 = model(images_h)
                probs_2 = torch.flip(torch.sigmoid(logits_2), [3])

                # 3. Vertical Flip (dim 2)
                images_v = torch.flip(images, [2])
                logits_3 = model(images_v)
                probs_3 = torch.flip(torch.sigmoid(logits_3), [2])

                # 4. Rotate 180 (H + V Flip)
                images_hv = torch.flip(images, [2, 3])
                logits_4 = model(images_hv)
                probs_4 = torch.flip(torch.sigmoid(logits_4), [2, 3])

                # Average probabilities
                avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0
            else:
                # No TTA
                logits = model(images)
                avg_probs = torch.sigmoid(logits)

            # 5. Post-Processing & Encoding
            # Convert to binary mask
            preds = (avg_probs > threshold).float().cpu().numpy()

            # Iterate over batch to encode
            for i, record_id in enumerate(record_ids):
                # Pred shape is (1, H, W), we need (H, W) for RLE
                mask = preds[i, 0, :, :]

                encoded_string = rle_encode(mask)
                results.append(
                    {"record_id": record_id, "encoded_pixels": encoded_string}
                )

    # 6. Save Submission
    df_submission = pd.DataFrame(results)

    # Ensure columns are in correct order
    df_submission = df_submission[["record_id", "encoded_pixels"]]

    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(
        f"Submission saved to {Config.SUBMISSION_PATH} with {len(df_submission)} records."
    )
