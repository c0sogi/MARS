import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Configures cuDNN for deterministic execution.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_rmse(y_pred, y_true):
    """
    Calculates the Root Mean Squared Error (RMSE) between predicted and true values.

    Args:
        y_pred (np.ndarray or torch.Tensor): Predicted values.
        y_true (np.ndarray or torch.Tensor): Ground truth values.

    Returns:
        float: The RMSE value.
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    mse = np.mean((y_pred - y_true) ** 2)
    return np.sqrt(mse)


def tiled_inference(
    model,
    image,
    patch_size=Config.PATCH_SIZE,
    overlap_ratio=Config.OVERLAP_RATIO,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
):
    """
    Performs inference on a large image by tiling it into overlapping patches.
    Handles padding, batch processing, and reconstruction with averaging in overlapping regions.

    Args:
        model (torch.nn.Module): The trained model.
        image (torch.Tensor): Input image tensor of shape (1, H, W) or (C, H, W).
        patch_size (int): Size of the square patches.
        overlap_ratio (float): Fraction of overlap between patches (0.0 to 1.0).
        batch_size (int): Number of patches to process in a single batch.
        device (str): Device to run inference on.

    Returns:
        torch.Tensor: Reconstructed output tensor of shape (C, H, W).
    """
    model.eval()

    # Ensure image is 4D: (1, C, H, W)
    if image.dim() == 3:
        image = image.unsqueeze(0)

    b, c, h, w = image.shape

    # Calculate stride based on overlap
    stride = int(patch_size * (1 - overlap_ratio))

    # Calculate padding to ensure the image is fully covered by patches
    # We ensure that (H_pad - patch_size) % stride == 0
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride

    # Handle edge case where image is smaller than patch_size
    if h < patch_size:
        pad_h += patch_size - h
    if w < patch_size:
        pad_w += patch_size - w

    # Pad the image. Pad format is (left, right, top, bottom)
    # Using reflect padding to minimize edge artifacts
    image_pad = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

    h_pad, w_pad = image_pad.shape[2], image_pad.shape[3]

    # Extract patches using unfold
    # Input: (B, C, H_pad, W_pad) -> Output: (B, C*kernel*kernel, L)
    patches = F.unfold(image_pad, kernel_size=patch_size, stride=stride)

    # Reshape for model input: (B, C*K*K, L) -> (B, L, C*K*K) -> (B*L, C, K, K)
    num_patches = patches.shape[2]
    patches = patches.permute(0, 2, 1).contiguous()
    patches = patches.view(num_patches, c, patch_size, patch_size)

    # Process patches in batches to manage memory
    output_patches_list = []

    with torch.no_grad():
        for i in range(0, num_patches, batch_size):
            batch = patches[i : i + batch_size].to(device)

            # Forward pass
            # The model predicts the output for the patch (e.g., noise residual)
            pred = model(batch)

            output_patches_list.append(pred.cpu())

    # Concatenate all output patches: (L, C, K, K)
    output_patches = torch.cat(output_patches_list, dim=0)

    # Reshape back for fold: (1, C*K*K, L)
    # Note: Assuming model output channels == input channels (C)
    output_patches = output_patches.view(1, num_patches, c * patch_size * patch_size)
    output_patches = output_patches.permute(0, 2, 1)

    # Fold patches back to full image size
    # Output: (1, C, H_pad, W_pad)
    output_pad = F.fold(
        output_patches,
        output_size=(h_pad, w_pad),
        kernel_size=patch_size,
        stride=stride,
    )

    # Create a weight map to handle overlaps (averaging)
    # We fold a tensor of ones to count how many times each pixel was predicted
    ones = torch.ones_like(image_pad)
    ones_patches = F.unfold(ones, kernel_size=patch_size, stride=stride)
    ones_fold = F.fold(
        ones_patches, output_size=(h_pad, w_pad), kernel_size=patch_size, stride=stride
    )

    # Normalize by the count of overlaps
    output_pad /= ones_fold.to(output_pad.device)

    # Crop back to original size
    output = output_pad[:, :, :h, :w]

    return output.squeeze(0)  # Returns (C, H, W)


def apply_tta(
    model,
    image,
    patch_size=Config.PATCH_SIZE,
    overlap_ratio=Config.OVERLAP_RATIO,
    device=Config.DEVICE,
):
    """
    Applies Test-Time Augmentation (TTA) by averaging predictions from augmented versions of the input.
    Augmentations: Identity, Horizontal Flip, Vertical Flip, Rotate 90, Rotate 270.

    Args:
        model (torch.nn.Module): The trained model.
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        patch_size (int): Patch size for tiled inference.
        overlap_ratio (float): Overlap ratio for tiled inference.
        device (str): Device for inference.

    Returns:
        torch.Tensor: Averaged prediction tensor.
    """
    preds = []

    # 1. Identity
    pred_raw = tiled_inference(
        model,
        image,
        patch_size,
        overlap_ratio,
        batch_size=Config.BATCH_SIZE,
        device=device,
    )
    preds.append(pred_raw)

    # 2. Horizontal Flip
    img_hflip = torch.flip(image, dims=[-1])
    out_hflip = tiled_inference(
        model,
        img_hflip,
        patch_size,
        overlap_ratio,
        batch_size=Config.BATCH_SIZE,
        device=device,
    )
    preds.append(torch.flip(out_hflip, dims=[-1]))

    # 3. Vertical Flip
    img_vflip = torch.flip(image, dims=[-2])
    out_vflip = tiled_inference(
        model,
        img_vflip,
        patch_size,
        overlap_ratio,
        batch_size=Config.BATCH_SIZE,
        device=device,
    )
    preds.append(torch.flip(out_vflip, dims=[-2]))

    # 4. Rotate 90 degrees (k=1)
    # rot90 rotates in the plane defined by dims. For (C, H, W), dims=[-2, -1] are H and W.
    img_rot90 = torch.rot90(image, k=1, dims=[-2, -1])
    out_rot90 = tiled_inference(
        model,
        img_rot90,
        patch_size,
        overlap_ratio,
        batch_size=Config.BATCH_SIZE,
        device=device,
    )
    # Inverse is rot270 (k=3)
    preds.append(torch.rot90(out_rot90, k=3, dims=[-2, -1]))

    # 5. Rotate 270 degrees (k=3)
    img_rot270 = torch.rot90(image, k=3, dims=[-2, -1])
    out_rot270 = tiled_inference(
        model,
        img_rot270,
        patch_size,
        overlap_ratio,
        batch_size=Config.BATCH_SIZE,
        device=device,
    )
    # Inverse is rot90 (k=1)
    preds.append(torch.rot90(out_rot270, k=1, dims=[-2, -1]))

    # Average predictions
    avg_pred = torch.stack(preds).mean(dim=0)

    return avg_pred
