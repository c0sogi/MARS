import os
import cv2
import numpy as np
import pandas as pd
import torch
from library import config, utils, network


def predict_with_tta(model, image, device):
    """
    Predicts noise for a full image using Geometric Self-Ensemble (TTA).

    Args:
        model (torch.nn.Module): The trained model.
        image (np.ndarray): Normalized input image (H, W).
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Averaged noise prediction (H, W).
    """
    # Prepare input tensor: (1, 1, H, W)
    img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)

    # If TTA is disabled in config, perform single pass
    if not config.USE_TTA:
        model.eval()
        with torch.no_grad():
            noise_pred = model(img_tensor)
        return noise_pred.squeeze().cpu().numpy()

    # Define transforms (D4 Dihedral Group)
    # 0: Identity
    # 1: Rot90
    # 2: Rot180
    # 3: Rot270
    # 4: Flip Horizontal
    # 5: Flip H + Rot90
    # 6: Flip H + Rot180
    # 7: Flip H + Rot270

    transforms = [
        lambda x: x,
        lambda x: torch.rot90(x, 1, [2, 3]),
        lambda x: torch.rot90(x, 2, [2, 3]),
        lambda x: torch.rot90(x, 3, [2, 3]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.rot90(torch.flip(x, [3]), 1, [2, 3]),
        lambda x: torch.rot90(torch.flip(x, [3]), 2, [2, 3]),
        lambda x: torch.rot90(torch.flip(x, [3]), 3, [2, 3]),
    ]

    # Define inverse transforms
    # Note: Inverse of (Rot * Flip) is (Flip * Rot_inv)
    inverse_transforms = [
        lambda x: x,
        lambda x: torch.rot90(x, -1, [2, 3]),
        lambda x: torch.rot90(x, -2, [2, 3]),
        lambda x: torch.rot90(x, -3, [2, 3]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(torch.rot90(x, -1, [2, 3]), [3]),
        lambda x: torch.flip(torch.rot90(x, -2, [2, 3]), [3]),
        lambda x: torch.flip(torch.rot90(x, -3, [2, 3]), [3]),
    ]

    noise_accum = torch.zeros_like(img_tensor)
    model.eval()

    with torch.no_grad():
        for t, inv_t in zip(transforms, inverse_transforms):
            # Augment
            aug_img = t(img_tensor)

            # Predict
            pred_noise = model(aug_img)

            # Inverse Augment
            pred_noise = inv_t(pred_noise)

            # Accumulate
            noise_accum += pred_noise

    # Average
    noise_avg = noise_accum / len(transforms)
    return noise_avg.squeeze().cpu().numpy()


def generate_submission(
    model_path=config.BEST_MODEL_PATH, output_path=config.SUBMISSION_FILE
):
    """
    Generates the submission file for the test set.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
    """
    print(f"Generating submission from model: {model_path}")

    device = torch.device(config.DEVICE)

    # 1. Initialize Model
    model = network.SE_ZI_ResDnCNN(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        num_features=config.NUM_FEATURES,
        num_blocks=config.NUM_BLOCKS,
        kernel_size=config.KERNEL_SIZE,
        padding=config.PADDING,
        use_se=config.USE_SE,
        se_reduction=config.SE_REDUCTION,
        zero_init_residual=config.ZERO_INIT_RESIDUAL,
    ).to(device)

    # 2. Load Weights
    start_epoch, best_metric = utils.load_checkpoint(model_path, model, device=device)
    print(
        f"Model loaded. Training stopped at epoch {start_epoch} with validation metric {best_metric}"
    )

    # 3. Load Test Metadata
    if not os.path.exists(config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {config.TEST_METADATA_PATH}"
        )

    df_test = pd.read_csv(config.TEST_METADATA_PATH)
    print(f"Found {len(df_test)} test images.")

    submission_dfs = []

    # 4. Process Each Image
    for idx, row in df_test.iterrows():
        img_id = row["image_id"]
        input_path = os.path.join(config.INPUT_DIR, row["input_path"])

        # Read Image
        img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img_in is None:
            print(f"Warning: Could not read image {input_path}")
            continue

        h, w = img_in.shape

        # Normalize [0, 1]
        img_norm = utils.normalize_image(img_in)

        # Predict Noise
        noise_pred = predict_with_tta(model, img_norm, device)

        # Reconstruct Clean Image: Input - Noise
        clean_pred = img_norm - noise_pred

        # Clip to valid range [0, 1]
        clean_pred = np.clip(clean_pred, 0.0, 1.0)

        # 5. Format for Submission
        # Format: id={image_id_no_ext}_{row}_{col}, value={intensity}
        # Note: Row and Col are 1-based indices

        base_id = os.path.splitext(img_id)[0]

        # Create coordinate grids
        # rows: 1..h, cols: 1..w
        # We repeat row indices for each column, and tile column indices for each row
        row_indices = np.repeat(np.arange(1, h + 1), w)
        col_indices = np.tile(np.arange(1, w + 1), h)

        # Flatten pixel values (row-major order matches the repeat/tile strategy)
        flat_values = clean_pred.flatten()

        # Construct ID strings
        # Vectorized string formatting is tricky in numpy, using list comprehension is safer/easier
        # Given the size, this loop is the bottleneck, but unavoidable for string generation
        ids = [f"{base_id}_{r}_{c}" for r, c in zip(row_indices, col_indices)]

        df_img = pd.DataFrame({"id": ids, "value": flat_values})

        submission_dfs.append(df_img)

        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(df_test)} images...")

    # 6. Concatenate and Save
    if submission_dfs:
        full_submission = pd.concat(submission_dfs, ignore_index=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        full_submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path} with {len(full_submission)} rows.")
    else:
        print("Error: No predictions generated.")
