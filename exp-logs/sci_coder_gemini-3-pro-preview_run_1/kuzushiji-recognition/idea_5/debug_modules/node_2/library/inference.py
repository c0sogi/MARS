import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.model import SwinCenterNet
from library.dataset import KuzushijiDataset
from library.utils import decode_center_net


def generate_submission(weights_path=None, debug_size=None):
    """
    Generates the submission file for the test dataset.

    Args:
        weights_path (str, optional): Path to the trained model weights.
                                      Defaults to Config.BEST_MODEL_PATH.
        debug_size (int, optional): Number of samples to process for debugging.
                                    Defaults to None (process all).
    """
    # 1. Setup Configuration and Device
    device = Config.DEVICE
    if weights_path is None:
        weights_path = Config.BEST_MODEL_PATH

    print(f"Starting inference using device: {device}")

    # 2. Load Unicode Mapping (Index -> Character)
    # We need to map the predicted class ID back to the Unicode string.
    try:
        df_uni = pd.read_csv(Config.UNICODE_MAP_PATH)
        if "Unicode" in df_uni.columns:
            chars = df_uni["Unicode"].values
        else:
            # Fallback if column name differs, usually it's the first column
            chars = df_uni.iloc[:, 0].values
        idx_to_char = {i: c for i, c in enumerate(chars)}
    except FileNotFoundError:
        print(f"Error: Unicode map file not found at {Config.UNICODE_MAP_PATH}")
        return

    # 3. Initialize Model
    model = SwinCenterNet()

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {weights_path}")
    else:
        print(
            f"Warning: Weights file not found at {weights_path}. Using random initialization."
        )

    model.to(device)
    model.eval()

    # 4. Prepare Data Loader
    test_dataset = KuzushijiDataset(split="test", debug_size=debug_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    results = []

    # 5. Inference Loop
    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            # orig_size is (B, 2) -> [Height, Width]
            orig_sizes = batch["orig_size"].numpy()

            # Forward Pass
            outputs = model(images)

            # Decode Predictions
            # preds shape: (B, K, 6) -> [x, y, w, h, score, class]
            # x, y are centers in the feature map coordinate system (Output Stride)
            preds = decode_center_net(
                outputs["hm"], outputs["wh"], outputs["reg"], K=Config.MAX_DETECTIONS
            )

            preds = preds.cpu().numpy()

            # Process each image in the batch
            for i, img_id in enumerate(image_ids):
                p_det = preds[i]

                # Filter by Confidence Threshold
                valid_mask = p_det[:, 4] >= Config.CONF_THRESHOLD
                p_det = p_det[valid_mask]

                # Sort by Score (Descending)
                if len(p_det) > 0:
                    p_det = p_det[np.argsort(-p_det[:, 4])]

                # Enforce Max Detections Limit
                if len(p_det) > Config.MAX_DETECTIONS:
                    p_det = p_det[: Config.MAX_DETECTIONS]

                # Calculate Scaling Factors
                # Model input is 1024x1024. Feature map is 1024/stride.
                # We need to map from Feature Map -> 1024x1024 -> Original Image

                orig_h, orig_w = orig_sizes[i]

                # Scale from 1024x1024 to Original Image
                scale_x = orig_w / Config.IMG_SIZE[1]
                scale_y = orig_h / Config.IMG_SIZE[0]

                label_strs = []

                for det in p_det:
                    # det: [x_feat, y_feat, w_feat, h_feat, score, class]
                    x_feat = det[0]
                    y_feat = det[1]
                    cls_idx = int(det[5])

                    # 1. Scale from Feature Map to Model Input Size (1024x1024)
                    x_1024 = x_feat * Config.OUTPUT_STRIDE
                    y_1024 = y_feat * Config.OUTPUT_STRIDE

                    # 2. Scale from Model Input Size to Original Image Size
                    final_x = int(x_1024 * scale_x)
                    final_y = int(y_1024 * scale_y)

                    # Retrieve Character
                    if cls_idx in idx_to_char:
                        char = idx_to_char[cls_idx]
                        # Format: Unicode X Y
                        label_strs.append(f"{char} {final_x} {final_y}")

                # Join all predictions for this image
                labels_str = " ".join(label_strs)
                results.append({"image_id": img_id, "labels": labels_str})

    # 6. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df = pd.DataFrame(results)

    # Ensure columns are in correct order
    sub_df = sub_df[["image_id", "labels"]]

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
