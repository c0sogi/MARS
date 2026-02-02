import os
import cv2
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.model import ZIResDnCNN


def apply_tta(model, x):
    """
    Applies 8-way Geometric Self-Ensemble (Test-Time Augmentation).
    Averages predictions from original and geometric transformations.

    Args:
        model: The PyTorch model.
        x: Input tensor (1, 1, H, W).

    Returns:
        Averaged prediction tensor.
    """
    # List to store predictions
    preds = []

    # 1. Original
    preds.append(model(x))

    # 2. Rot90 (k=1)
    x_rot1 = torch.rot90(x, 1, [2, 3])
    out_rot1 = model(x_rot1)
    preds.append(torch.rot90(out_rot1, 3, [2, 3]))

    # 3. Rot180 (k=2)
    x_rot2 = torch.rot90(x, 2, [2, 3])
    out_rot2 = model(x_rot2)
    preds.append(torch.rot90(out_rot2, 2, [2, 3]))

    # 4. Rot270 (k=3)
    x_rot3 = torch.rot90(x, 3, [2, 3])
    out_rot3 = model(x_rot3)
    preds.append(torch.rot90(out_rot3, 1, [2, 3]))

    # 5. FlipLR (Horizontal Flip)
    # Tensor is (B, C, H, W). Flip dim 3 is Width (Horizontal).
    x_flip = torch.flip(x, [3])
    out_flip = model(x_flip)
    preds.append(torch.flip(out_flip, [3]))

    # 6. FlipLR + Rot90
    x_flip_rot1 = torch.rot90(x_flip, 1, [2, 3])
    out_flip_rot1 = model(x_flip_rot1)
    # Inverse: Flip(Rot_inv(y)) -> Flip(Rot270(y))
    preds.append(torch.flip(torch.rot90(out_flip_rot1, 3, [2, 3]), [3]))

    # 7. FlipLR + Rot180
    x_flip_rot2 = torch.rot90(x_flip, 2, [2, 3])
    out_flip_rot2 = model(x_flip_rot2)
    # Inverse: Flip(Rot180(y))
    preds.append(torch.flip(torch.rot90(out_flip_rot2, 2, [2, 3]), [3]))

    # 8. FlipLR + Rot270
    x_flip_rot3 = torch.rot90(x_flip, 3, [2, 3])
    out_flip_rot3 = model(x_flip_rot3)
    # Inverse: Flip(Rot90(y))
    preds.append(torch.flip(torch.rot90(out_flip_rot3, 1, [2, 3]), [3]))

    # Stack and calculate mean
    return torch.stack(preds).mean(dim=0)


def run_inference(
    checkpoint_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_PATH,
    metadata_path=Config.TEST_METADATA_PATH,
    use_tta=Config.USE_TTA,
    device=Config.DEVICE,
):
    """
    Runs the inference pipeline: loads model, predicts on test images, and generates submission CSV.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        metadata_path (str): Path to the test metadata CSV.
        use_tta (bool): Whether to use Test-Time Augmentation.
        device (str): Device to run inference on ('cuda' or 'cpu').
    """
    set_seed(Config.SEED)
    device = torch.device(device)

    print(f"Initializing Inference on {device}...")

    # 1. Initialize Model
    model = ZIResDnCNN(
        num_blocks=Config.NUM_BLOCKS,
        num_channels=Config.NUM_CHANNELS,
        kernel_size=Config.KERNEL_SIZE,
        padding=Config.PADDING,
        use_zero_gamma=Config.USE_ZERO_GAMMA,
    ).to(device)

    # 2. Load Checkpoint
    try:
        load_checkpoint(model, filename=checkpoint_path)
        print(f"Model loaded successfully from {checkpoint_path}")
    except FileNotFoundError:
        print(
            f"Error: Checkpoint not found at {checkpoint_path}. Ensure training is complete."
        )
        return

    model.eval()

    # 3. Load Test Metadata
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    df_test = pd.read_csv(metadata_path)
    print(f"Found {len(df_test)} test images.")

    # 4. Processing Loop
    results = []

    with torch.no_grad():
        for idx, row in df_test.iterrows():
            img_id_full = row["image_id"]
            # Remove extension for ID format (e.g., "110.png" -> "110")
            img_id = os.path.splitext(img_id_full)[0]

            input_rel_path = row["input_path"]
            input_full_path = os.path.join(Config.INPUT_DIR, input_rel_path)

            # Load Image
            img = cv2.imread(input_full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {input_full_path}. Skipping.")
                continue

            # Preprocess
            # Normalize to [0, 1]
            img_norm = img.astype(np.float32) / 255.0

            # Convert to Tensor (1, 1, H, W)
            input_tensor = (
                torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
            )

            # Predict
            if use_tta:
                output_tensor = apply_tta(model, input_tensor)
            else:
                output_tensor = model(input_tensor)

            # Post-process
            # Move to CPU, squeeze dims
            output_np = output_tensor.squeeze().cpu().numpy()

            # Clip values to valid range [0, 1]
            output_np = np.clip(output_np, 0, 1)

            # Flatten and format for submission
            h, w = output_np.shape

            # Generate 1-based indices for rows and columns
            # np.indices returns (2, h, w) array of indices
            # r_indices[y, x] = y, c_indices[y, x] = x
            r_indices, c_indices = np.indices((h, w))
            r_indices += 1  # 1-based
            c_indices += 1  # 1-based

            # Flatten everything
            r_flat = r_indices.flatten()
            c_flat = c_indices.flatten()
            val_flat = output_np.flatten()

            # Generate ID strings: "imageID_row_col"
            # Using list comprehension is efficient enough for this scale
            id_list = [f"{img_id}_{r}_{c}" for r, c in zip(r_flat, c_flat)]

            # Store in a temporary DataFrame
            df_img = pd.DataFrame({"id": id_list, "value": val_flat})

            results.append(df_img)

            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{len(df_test)} images...")

    # 5. Save Submission
    if results:
        print("Concatenating results...")
        final_df = pd.concat(results, ignore_index=True)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Saving submission file to {output_path}...")
        final_df.to_csv(output_path, index=False)
        print("Submission generation complete.")
    else:
        print("No results generated.")
