import os
import torch
import numpy as np
import pandas as pd
from library.config import Config


def predict_tiled(
    model,
    image,
    patch_size=Config.PATCH_SIZE,
    overlap=Config.INFERENCE_OVERLAP,
    device=Config.DEVICE,
):
    """
    Performs inference on a large image using a sliding window approach with overlap.

    Args:
        model (nn.Module): The trained model.
        image (torch.Tensor): Input image tensor of shape (1, C, H, W).
        patch_size (int): Size of the square patch.
        overlap (float): Overlap ratio between patches (0 to 1).
        device (str or torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Denoised image tensor of shape (1, C, H, W).
    """
    model.eval()
    b, c, h, w = image.shape

    # Calculate stride
    stride = int(patch_size * (1 - overlap))

    # Initialize accumulators
    output = torch.zeros((b, c, h, w), device=device)
    count = torch.zeros((b, c, h, w), device=device)

    # Generate coordinates
    # We ensure we cover the edges by explicitly adding the last possible position
    y_coords = list(range(0, h - patch_size + 1, stride))
    if (h - patch_size) % stride != 0:
        y_coords.append(h - patch_size)
    # Ensure the last patch is always included if the loop didn't reach the end perfectly
    if y_coords[-1] != h - patch_size:
        y_coords.append(h - patch_size)

    x_coords = list(range(0, w - patch_size + 1, stride))
    if (w - patch_size) % stride != 0:
        x_coords.append(w - patch_size)
    if x_coords[-1] != w - patch_size:
        x_coords.append(w - patch_size)

    # Remove duplicates just in case
    y_coords = sorted(list(set(y_coords)))
    x_coords = sorted(list(set(x_coords)))

    with torch.no_grad():
        for y in y_coords:
            for x in x_coords:
                # Extract patch
                patch = image[:, :, y : y + patch_size, x : x + patch_size].to(device)

                # Inference
                pred_patch = model(patch)

                # Accumulate
                output[:, :, y : y + patch_size, x : x + patch_size] += pred_patch
                count[:, :, y : y + patch_size, x : x + patch_size] += 1.0

    # Average overlapping regions
    output = output / count
    return output


def predict_tta(
    model,
    image,
    patch_size=Config.PATCH_SIZE,
    overlap=Config.INFERENCE_OVERLAP,
    device=Config.DEVICE,
):
    """
    Performs Test-Time Augmentation (TTA) by averaging predictions of transformed versions.

    Transforms used:
    1. Identity
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotate 90 degrees (k=1)

    Args:
        model (nn.Module): The trained model.
        image (torch.Tensor): Input image tensor.
        patch_size (int): Patch size.
        overlap (float): Overlap ratio.
        device (str or torch.device): Device.

    Returns:
        torch.Tensor: Averaged denoised image.
    """
    model.eval()
    preds = []

    # 1. Original
    pred_orig = predict_tiled(model, image, patch_size, overlap, device)
    preds.append(pred_orig)

    # 2. Horizontal Flip
    img_h = torch.flip(image, [3])
    pred_h = predict_tiled(model, img_h, patch_size, overlap, device)
    preds.append(torch.flip(pred_h, [3]))

    # 3. Vertical Flip
    img_v = torch.flip(image, [2])
    pred_v = predict_tiled(model, img_v, patch_size, overlap, device)
    preds.append(torch.flip(pred_v, [2]))

    # 4. Rotate 90 (k=1)
    # rot90 rotates dims (2, 3).
    img_r90 = torch.rot90(image, k=1, dims=[2, 3])
    pred_r90 = predict_tiled(model, img_r90, patch_size, overlap, device)
    # Inverse is rot90 k=3 (or -1)
    preds.append(torch.rot90(pred_r90, k=3, dims=[2, 3]))

    # Average predictions
    final_pred = torch.stack(preds).mean(dim=0)
    return final_pred


def generate_submission_inference(model, dataloader, device, output_path):
    """
    Generates the submission file using TTA and Tiled Inference.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test data loader.
        device (torch.device): Device.
        output_path (str): Path to save the CSV.
    """
    model.eval()
    results = []

    print("Generating submission with TTA and Tiled Inference...")

    for inputs, _, img_ids in dataloader:
        # Inputs are (B, C, H, W)
        # Note: The dataloader for test returns full images.
        # Batch size is likely 1 for varying image sizes, or >1 if resized (but we don't resize).
        # Assuming batch size 1 for safety with varying sizes, or consistent sizes.

        for i in range(len(img_ids)):
            img_id = img_ids[i]
            img_tensor = inputs[i : i + 1]  # Keep 4D shape (1, C, H, W)

            # Run TTA Inference
            output_tensor = predict_tta(model, img_tensor, device=device)

            # Clamp to [0, 1]
            output_tensor = torch.clamp(output_tensor, 0, 1)

            # Convert to numpy
            # Shape (1, 1, H, W) -> (H, W)
            img_pred = output_tensor.cpu().numpy()[0, 0, :, :]
            h, w = img_pred.shape

            # Create coordinate grids (1-based indexing)
            grid = np.indices((h, w))
            rows = grid[0].flatten() + 1
            cols = grid[1].flatten() + 1
            vals = img_pred.flatten()

            # Create IDs
            ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows, cols)]

            # Create DataFrame
            df_img = pd.DataFrame({"id": ids, "value": vals})
            results.append(df_img)

    # Concatenate and save
    if results:
        final_df = pd.concat(results, ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    else:
        print("No results generated.")
