import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import (
    process_and_cache_25d_metadata,
    GI_MRI_Dataset,
    get_transforms,
)
from library.model import FPN


def run_inference(
    model_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
    batch_size=Config.BATCH_SIZE,
    debug=False,
):
    """
    Runs the inference pipeline: loads data, loads model, generates predictions,
    resizes to original resolution, encodes masks, and saves submission file.

    Args:
        model_path (str): Path to the trained model weights.
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs on a small subset of the test data.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    # 1. Setup
    Config.setup(debug=debug, training=False)
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting inference on device: {device}")

    # 2. Load Metadata
    # We only need the test dataframe here.
    # process_and_cache_25d_metadata handles loading/generating the neighbor paths.
    _, _, test_df = process_and_cache_25d_metadata(load_cached_data=True)

    if debug:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} test rows.")
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Create a lookup for original image dimensions: id -> (height, width)
    # This is necessary to resize the fixed-size model output back to the original scan size.
    dim_lookup = test_df.set_index("id")[["height", "width"]].to_dict("index")

    # 3. Prepare Dataset and Loader
    test_dataset = GI_MRI_Dataset(
        df=test_df, transforms=get_transforms(mode="test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model
    model = FPN(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # Weights are loaded from checkpoint, no need to download ImageNet weights
        num_classes=Config.NUM_CLASSES,
    )

    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model path {model_path} not found. Inference will use random weights."
        )

    model.to(device)
    model.eval()

    results = []

    # 5. Inference Loop
    print("Running prediction loop...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Forward pass
            # Output shape: (Batch, Num_Classes, H_model, W_model) -> (B, 3, 320, 320)
            preds = model(images)

            # Move to CPU for post-processing
            preds = preds.cpu().numpy()

            # Process each sample in the batch
            for i, slice_id in enumerate(ids):
                # Retrieve original dimensions
                orig_h = dim_lookup[slice_id]["height"]
                orig_w = dim_lookup[slice_id]["width"]

                # Get prediction for current sample: (3, 320, 320)
                pred_vol = preds[i]

                # Transpose to (320, 320, 3) for cv2.resize which expects (H, W, C)
                pred_vol = np.transpose(pred_vol, (1, 2, 0))

                # Resize to original resolution (width, height)
                # Note: cv2.resize takes (width, height) as destination size
                pred_resized = cv2.resize(
                    pred_vol, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )

                # Threshold probabilities to get binary mask
                pred_mask = (pred_resized > 0.5).astype(
                    np.uint8
                )  # Shape: (Orig_H, Orig_W, 3)

                # Encode masks for each class
                for cls_idx, cls_name in enumerate(Config.CLASSES):
                    # Extract single class mask
                    cls_mask = pred_mask[:, :, cls_idx]

                    # Run-Length Encoding
                    rle = rle_encode(cls_mask)

                    results.append(
                        {"id": slice_id, "class": cls_name, "predicted": rle}
                    )

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure correct column order
    submission_df = submission_df[["id", "class", "predicted"]]

    # Save to disk
    out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(out_path, index=False)

    print(f"Inference complete. Submission saved to {out_path}")
    print(f"Total rows in submission: {len(submission_df)}")

    return submission_df
