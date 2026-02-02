import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import CFG
from library.utils import set_seed, rle_encode
from library.dataset import ContrailDataset, get_transforms
from library.model import ConvNeXtUNet


def predict_and_submit(debug=False):
    """
    Loads the trained model, performs inference on the test set using Test-Time Augmentation (TTA),
    and generates the submission file in Run-Length Encoding (RLE) format.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data for debugging.
    """
    # Ensure reproducibility across runs
    set_seed(CFG.seed)

    print(f"Starting Inference Pipeline (Debug={debug})...")

    # ====================================================
    # Data Loading
    # ====================================================
    # Initialize the test dataset
    # The dataset class handles caching of processed inputs automatically
    test_dataset = ContrailDataset(
        metadata_path=CFG.test_metadata_path,
        split="test",
        transform=get_transforms("test"),  # Applies ToTensorV2 only
        debug=debug,
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Test Dataset Size: {len(test_dataset)}")

    # ====================================================
    # Model Initialization
    # ====================================================
    model = ConvNeXtUNet(
        backbone_name=CFG.backbone,
        in_channels=CFG.in_channels,
        num_classes=CFG.out_channels,
        pretrained=False,  # We are loading custom weights, so ImageNet weights are not needed
    )

    # Load the best trained weights
    if os.path.exists(CFG.best_model_path):
        print(f"Loading model weights from {CFG.best_model_path}...")
        state_dict = torch.load(CFG.best_model_path, map_location=CFG.device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model checkpoint not found at {CFG.best_model_path}. Predictions will be random."
        )

    model.to(CFG.device)
    model.eval()

    # ====================================================
    # Inference Loop
    # ====================================================
    results = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(CFG.device, dtype=torch.float32)
            record_ids = batch["record_id"]

            # --- Test-Time Augmentation (TTA) ---

            # 1. Original View
            pred_orig = model(images)

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            pred_h = model(images_h)
            pred_h = torch.flip(pred_h, dims=[3])  # Flip back

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[2])
            pred_v = model(images_v)
            pred_v = torch.flip(pred_v, dims=[2])  # Flip back

            # 4. 180 Degree Rotation (Horizontal + Vertical Flip)
            images_rot = torch.flip(images, dims=[2, 3])
            pred_rot = model(images_rot)
            pred_rot = torch.flip(pred_rot, dims=[2, 3])  # Rotate back

            # Average predictions to reduce variance
            avg_logits = (pred_orig + pred_h + pred_v + pred_rot) / 4.0

            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(avg_logits)

            # Move to CPU for post-processing
            probs_np = probs.cpu().numpy()

            # Process each image in the batch
            for i, record_id in enumerate(record_ids):
                # Extract probability map for the current image (H, W)
                prob_map = probs_np[i, 0]

                # Thresholding
                mask = prob_map > CFG.threshold

                # Run-Length Encoding
                # Convert boolean mask to uint8 (0, 1) for encoding
                encoded_pixels = rle_encode(mask.astype(np.uint8))

                results.append(
                    {"record_id": record_id, "encoded_pixels": encoded_pixels}
                )

    # ====================================================
    # Submission Generation
    # ====================================================
    submission_df = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(CFG.submission_dir, exist_ok=True)

    # Save to CSV
    submission_df.to_csv(CFG.submission_path, index=False)

    print(f"Inference complete. Submission saved to {CFG.submission_path}")
    print(f"Total records processed: {len(submission_df)}")
