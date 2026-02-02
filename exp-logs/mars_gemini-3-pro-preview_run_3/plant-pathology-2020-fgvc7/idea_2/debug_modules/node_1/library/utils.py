import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from library.config import seed_everything, CLASS_LABELS


def calculate_class_weights(df, device="cpu"):
    """
    Computes inverse frequency class weights based on the training set distribution
    to address class imbalance.

    Args:
        df (pd.DataFrame): The training dataframe. Must contain 'stratify_label' or
                           one-hot encoded columns matching CLASS_LABELS.
        device (str or torch.device): The device to place the tensor on. Defaults to "cpu".

    Returns:
        torch.Tensor: A tensor of class weights in the order of CLASS_LABELS.
    """
    # Determine the target labels
    if "stratify_label" in df.columns:
        y = df["stratify_label"].values
    else:
        # Fallback: derive from one-hot columns
        # Ensure we only look at columns that are in CLASS_LABELS
        # This assumes the dataframe has these columns
        y = df[CLASS_LABELS].idxmax(axis=1).values

    # Compute class weights using sklearn
    # "balanced" mode: n_samples / (n_classes * np.bincount(y))
    # We explicitly provide 'classes' to ensure the order matches CLASS_LABELS
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array(CLASS_LABELS), y=y
    )

    # Convert to PyTorch tensor
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Move to the specified device
    if device:
        weight_tensor = weight_tensor.to(device)

    return weight_tensor
