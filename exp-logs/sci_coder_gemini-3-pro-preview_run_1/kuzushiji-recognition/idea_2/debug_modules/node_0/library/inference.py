import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import KuzushijiDataset
from library.model import HRNetCenterNet
from library.engine import decode_outputs
from library.utils import get_affine_transform, affine_transform


def generate_submission(model, device, debug=False):
    """
    Generates submission file for the test set.

    Args:
        model (nn.Module): The trained model.
        device (torch.device): Device to run inference on.
        debug (bool): If True, runs on a small subset of the test data for debugging.
    """
    print("Generating submission...")

    # Determine debug size
    debug_size = 100 if debug else None

    # Load Dataset
    test_dataset = KuzushijiDataset(split="test", debug_size=debug_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    _, id2char = Config.get_class_mappings()
    results = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            img_ids = batch["image_id"]

            # Forward pass
            hm, wh, reg = model(imgs)

            # Decode outputs (Heatmap -> Points)
            # Returns coordinates in the input scale (1024x1024)
            scores, clses, xs, ys = decode_outputs(hm, wh, reg, K=Config.MAX_PREDS)

            # Process batch
            for b in range(len(img_ids)):
                img_id = img_ids[b]

                # Load original image size for inverse transform
                # We need to read the original image to get its dimensions because
                # the model input was resized/padded to 1024x1024.
                path = os.path.join(Config.INPUT_DIR, "test_images", f"{img_id}.jpg")

                # Fallback check
                if not os.path.exists(path):
                    path = os.path.join(
                        Config.INPUT_DIR, "test_images", f"{img_id}.jpg"
                    )

                if os.path.exists(path):
                    orig_img = cv2.imread(path)
                    if orig_img is not None:
                        oh, ow = orig_img.shape[:2]
                    else:
                        oh, ow = Config.INPUT_SIZE, Config.INPUT_SIZE
                else:
                    # Fallback if file not found
                    oh, ow = Config.INPUT_SIZE, Config.INPUT_SIZE

                # Calculate inverse affine transform (1024 -> Original)
                trans_inv = get_affine_transform(
                    (oh, ow), Config.INPUT_SIZE, inverse=True
                )

                label_strs = []

                # Filter by confidence threshold
                valid_mask = scores[b] > Config.CONF_THRESHOLD
                v_scores = scores[b][valid_mask]
                v_clses = clses[b][valid_mask]
                v_xs = xs[b][valid_mask]
                v_ys = ys[b][valid_mask]

                for k in range(len(v_scores)):
                    # Transform point back to original image coordinates
                    pt = affine_transform([v_xs[k], v_ys[k]], trans_inv)

                    # Clip coordinates to image bounds
                    px = int(min(max(0, pt[0]), ow - 1))
                    py = int(min(max(0, pt[1]), oh - 1))

                    char_code = id2char[v_clses[k]]

                    # Format: "Unicode X Y"
                    label_strs.append(f"{char_code} {px} {py}")

                # Join all predictions for this image
                results.append({"image_id": img_id, "labels": " ".join(label_strs)})

    # Save Submission
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_inference(debug=False):
    """
    Main entry point for inference.
    Sets up config, loads model, and generates submission.

    Args:
        debug (bool): If True, runs on a small subset of data.
    """
    Config.setup()
    Config.seed_everything(Config.SEED)

    device = Config.DEVICE

    # Initialize Model
    model = HRNetCenterNet().to(device)

    # Load Weights
    weights_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(
            f"Warning: Model weights not found at {weights_path}. Running with random weights (or pre-trained backbone only)."
        )

    # Run Generation
    generate_submission(model, device, debug=debug)
