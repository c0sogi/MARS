import os
import numpy as np
import torch
from library.dataset import get_test_loader, ORIG_SIZE, TARGET_SIZE
from library.utils import rle_encode


def predict_and_submit(
    model, device, output_dir="submission", output_name="submission.csv"
):
    """
    Performs inference on the test set using the provided model, applies Test-Time Augmentation (TTA),
    and generates a submission CSV file.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        device (torch.device): The device to run inference on (CPU or CUDA).
        output_dir (str): Directory to save the submission file.
        output_name (str): Name of the submission file.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Load test data
    # We use the default batch size of 32 and num_workers=2 as established in dataset.py
    test_loader = get_test_loader(batch_size=32, num_workers=2, load_cached_data=True)

    # Calculate padding parameters to crop the 128x128 prediction back to 101x101
    # The model input was padded using reflection padding.
    pad_h = TARGET_SIZE - ORIG_SIZE
    pad_top = pad_h // 2
    pad_w = TARGET_SIZE - ORIG_SIZE
    pad_left = pad_w // 2

    all_rles = []
    ids = test_loader.dataset.ids
    current_idx = 0

    print("Starting inference on test set with TTA...")

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # --- Test-Time Augmentation (TTA) ---

            # 1. Forward pass with original images
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            # 2. Forward pass with horizontally flipped images
            # Flip dimension 3 (width)
            inputs_flip = torch.flip(inputs, dims=[3])
            outputs_flip = model(inputs_flip)
            probs_flip = torch.sigmoid(outputs_flip)

            # Flip the predictions back to original orientation
            probs_flip_back = torch.flip(probs_flip, dims=[3])

            # 3. Average the probabilities
            avg_probs = (probs + probs_flip_back) / 2.0

            # --- Post-Processing ---

            # 4. Center Crop
            # avg_probs shape is (Batch, Channels, Height, Width) -> (B, 1, 128, 128)
            # We slice to get the center 101x101 region
            avg_probs_cropped = avg_probs[
                :, 0, pad_top : pad_top + ORIG_SIZE, pad_left : pad_left + ORIG_SIZE
            ]

            # 5. Thresholding
            # Convert probabilities to binary mask (0 or 1)
            preds_binary = (avg_probs_cropped > 0.5).cpu().numpy().astype(np.uint8)

            # 6. RLE Encoding
            for i in range(batch_size):
                rle = rle_encode(preds_binary[i])
                img_id = ids[current_idx]
                all_rles.append(f"{img_id},{rle}")
                current_idx += 1

    # --- Save Submission ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    with open(output_path, "w") as f:
        f.write("id,rle_mask\n")
        f.write("\n".join(all_rles))

    print(f"Submission saved to {output_path}")
