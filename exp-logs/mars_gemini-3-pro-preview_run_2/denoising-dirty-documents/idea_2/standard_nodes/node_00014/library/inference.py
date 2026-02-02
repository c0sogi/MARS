import os
import torch
import numpy as np
from library.model import ResUNet, predict_tiled
from library.utils import save_submission_file


def predict_with_tta(model, image_tensor, patch_size=128, overlap=32, device="cuda"):
    """
    Performs inference with Test Time Augmentation (TTA) including flips and rotations.
    Averages predictions from:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotate 90 degrees (counter-clockwise)
    5. Rotate 270 degrees (counter-clockwise)

    Args:
        model (nn.Module): The trained model.
        image_tensor (torch.Tensor): Input image tensor of shape (C, H, W).
        patch_size (int): Size of patches for tiled inference.
        overlap (int): Overlap size for tiled inference.
        device (str): Device to run inference on.

    Returns:
        torch.Tensor: Averaged denoised image tensor of shape (C, H, W).
    """
    preds = []

    # 1. Original
    p1 = predict_tiled(
        model, image_tensor, patch_size=patch_size, overlap=overlap, device=device
    )
    preds.append(p1)

    # 2. Horizontal Flip
    img_h = torch.flip(image_tensor, [2])
    p2 = predict_tiled(
        model, img_h, patch_size=patch_size, overlap=overlap, device=device
    )
    preds.append(torch.flip(p2, [2]))

    # 3. Vertical Flip
    img_v = torch.flip(image_tensor, [1])
    p3 = predict_tiled(
        model, img_v, patch_size=patch_size, overlap=overlap, device=device
    )
    preds.append(torch.flip(p3, [1]))

    # 4. Rotate 90 (k=1)
    # dims (1, 2) corresponds to (H, W) for (C, H, W) tensor
    img_r90 = torch.rot90(image_tensor, 1, [1, 2])
    p4 = predict_tiled(
        model, img_r90, patch_size=patch_size, overlap=overlap, device=device
    )
    preds.append(torch.rot90(p4, -1, [1, 2]))

    # 5. Rotate 270 (k=3)
    img_r270 = torch.rot90(image_tensor, 3, [1, 2])
    p5 = predict_tiled(
        model, img_r270, patch_size=patch_size, overlap=overlap, device=device
    )
    preds.append(torch.rot90(p5, -3, [1, 2]))

    # Average predictions
    final_pred = torch.stack(preds).mean(dim=0)
    return final_pred


def run_inference(
    test_data,
    model_path,
    output_path="./submission/submission.csv",
    device="cuda",
    patch_size=128,
    overlap=32,
    use_tta=True,
):
    """
    Generates predictions for the test dataset and saves them to a CSV file.

    Args:
        test_data (list): List of dictionaries containing 'id' and 'noisy' image data.
        model_path (str): Path to the saved model weights.
        output_path (str): Path to save the submission CSV.
        device (str): Device to run inference on.
        patch_size (int): Patch size for tiled inference.
        overlap (int): Overlap size for tiled inference.
        use_tta (bool): Whether to use Test Time Augmentation.
    """
    print(f"Loading model from {model_path}...")
    model = ResUNet().to(device)

    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    predictions = []
    ids = []

    print(f"Starting inference on {len(test_data)} images...")

    for i, item in enumerate(test_data):
        img_id = item["id"]
        noisy_np = item["noisy"]

        # Preprocess: Ensure (C, H, W)
        # noisy_np is likely (H, W) or (H, W, C)
        if noisy_np.ndim == 2:
            # (H, W) -> (1, H, W)
            noisy_t = torch.from_numpy(noisy_np).unsqueeze(0).float()
        elif noisy_np.ndim == 3:
            # (H, W, C) -> (C, H, W)
            noisy_t = torch.from_numpy(noisy_np).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unexpected image shape: {noisy_np.shape}")

        with torch.no_grad():
            if use_tta:
                pred = predict_with_tta(
                    model,
                    noisy_t,
                    patch_size=patch_size,
                    overlap=overlap,
                    device=device,
                )
            else:
                pred = predict_tiled(
                    model,
                    noisy_t,
                    patch_size=patch_size,
                    overlap=overlap,
                    device=device,
                )

        # Post-process: (C, H, W) -> (H, W) or (H, W, C)
        pred_np = pred.cpu().numpy()

        # If (1, H, W), squeeze to (H, W)
        if pred_np.shape[0] == 1:
            pred_np = pred_np.squeeze(0)
        else:
            # (C, H, W) -> (H, W, C)
            pred_np = np.transpose(pred_np, (1, 2, 0))

        predictions.append(pred_np)
        ids.append(img_id)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_data)} images")

    print(f"Saving submission to {output_path}...")
    save_submission_file(predictions, ids, output_path)
    print("Submission saved successfully.")
