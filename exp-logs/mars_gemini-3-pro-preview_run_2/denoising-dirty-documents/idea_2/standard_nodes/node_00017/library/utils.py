import os
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission_file(predictions, image_ids, output_path):
    """
    Formats predictions into the required CSV format (pixel-wise) and saves it.

    The format requires melting the image into rows:
    id,value
    {image_id}_{row}_{col},pixel_intensity

    Row and Column indices are 1-based.

    Args:
        predictions (list of np.ndarray or torch.Tensor): List of denoised images.
                                                          Expected shape (H, W) or (H, W, 1).
        image_ids (list of str): List of image IDs corresponding to the predictions.
        output_path (str): File path to save the generated CSV.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_list = []

    for img_id, img in zip(image_ids, predictions):
        # Convert Torch Tensor to Numpy if necessary
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # Handle channel dimension if present (H, W, 1) -> (H, W)
        if img.ndim == 3:
            img = img.squeeze()

        # Clip values to valid range [0, 1] as per task description
        img = np.clip(img, 0, 1)

        h, w = img.shape
        flat_vals = img.flatten()

        # Generate 1-based indices for rows and columns
        # Flatten order is row-major (C-style): (1,1), (1,2)... (1,W), (2,1)...
        r_idxs = np.repeat(np.arange(1, h + 1), w)
        c_idxs = np.tile(np.arange(1, w + 1), h)

        # Create ID strings: "{image_id}_{row}_{col}"
        ids = [f"{img_id}_{r}_{c}" for r, c in zip(r_idxs, c_idxs)]

        # Create a DataFrame for this image
        df_part = pd.DataFrame({"id": ids, "value": flat_vals})
        df_list.append(df_part)

    # Concatenate all image dataframes
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(output_path, index=False)
    else:
        # Handle empty prediction case by creating an empty file with header
        with open(output_path, "w") as f:
            f.write("id,value\n")
