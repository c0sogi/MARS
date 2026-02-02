import os
import torch
import numpy as np
import pandas as pd
from library import config, utils, dataset, model as model_lib


def tta_inference(model, images):
    """
    Performs Test-Time Augmentation (TTA) by averaging predictions from:
    1. Original image
    2. Horizontal Flip
    3. Vertical Flip
    4. 180-degree Rotation (Horizontal + Vertical Flip)

    Args:
        model: The PyTorch model in eval mode.
        images: Batch of images (B, C, H, W).

    Returns:
        avg_preds: Averaged probability maps (B, 1, H, W).
    """
    # 1. Original
    logits_orig = model(images)
    probs_orig = torch.sigmoid(logits_orig)

    # 2. Horizontal Flip (dim 3)
    images_h = torch.flip(images, dims=[3])
    logits_h = model(images_h)
    probs_h = torch.sigmoid(logits_h)
    probs_h = torch.flip(probs_h, dims=[3])  # Flip back

    # 3. Vertical Flip (dim 2)
    images_v = torch.flip(images, dims=[2])
    logits_v = model(images_v)
    probs_v = torch.sigmoid(logits_v)
    probs_v = torch.flip(probs_v, dims=[2])  # Flip back

    # 4. 180 Rotation (Flip dims 2 and 3)
    images_180 = torch.flip(images, dims=[2, 3])
    logits_180 = model(images_180)
    probs_180 = torch.sigmoid(logits_180)
    probs_180 = torch.flip(probs_180, dims=[2, 3])  # Flip back

    # Average
    avg_preds = (probs_orig + probs_h + probs_v + probs_180) / 4.0

    return avg_preds


def make_submission(model_path, debug=False):
    """
    Generates predictions for the test set and saves the submission CSV.

    Args:
        model_path (str): Path to the trained model weights (.pth file).
        debug (bool): If True, runs on a subset of the test data.
    """
    # Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE

    print(f"Generating submission...")
    print(f"Model Path: {model_path}")
    print(f"Debug Mode: {debug}")

    # Create submission directory
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Load Model
    # We use config.PRETRAINED as defined in the library config
    model = model_lib.ConvNeXtUNet(
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        in_channels=config.MODEL_INPUT_CHANNELS,
        num_classes=1,
    )

    # Load weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    else:
        print(f"Error: Model weights not found at {model_path}")
        return

    model.to(device)
    model.eval()

    # Data Loader
    test_loader = dataset.get_dataloader(
        stage="test",
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        debug=debug,
    )

    results = []

    # Inference Loop
    print(f"Starting inference on {len(test_loader)} batches...")
    with torch.no_grad():
        for batch_idx, (images, record_ids) in enumerate(test_loader):
            images = images.to(device, dtype=torch.float32)

            # TTA Inference
            if config.USE_TTA:
                probs = tta_inference(model, images)
            else:
                logits = model(images)
                probs = torch.sigmoid(logits)

            # Thresholding
            preds = (probs > config.THRESHOLD).float()

            # Move to CPU for encoding
            preds_np = preds.cpu().numpy()

            # Encode each image in batch
            for i in range(len(record_ids)):
                rid = record_ids[i]
                mask = preds_np[i, 0, :, :]  # (H, W)

                rle = utils.rle_encode(mask)
                results.append({"record_id": rid, "encoded_pixels": rle})

            # Periodic status update
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Save
    save_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Total records: {len(submission_df)}")
