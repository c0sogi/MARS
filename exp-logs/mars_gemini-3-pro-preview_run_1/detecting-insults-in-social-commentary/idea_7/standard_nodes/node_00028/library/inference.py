import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional

from library.config import Config
from library.data_processing import create_dataloader, get_tokenizer
from library.model import inference as model_inference


def predict_fn(
    model: torch.nn.Module,
    df: pd.DataFrame,
    svd_features: np.ndarray,
    device: torch.device,
    batch_size: Optional[int] = None,
    tokenizer=None,
) -> np.ndarray:
    """
    Generates probability predictions for a given dataset using the provided model.

    Args:
        model (torch.nn.Module): The trained model.
        df (pd.DataFrame): The dataframe containing text data.
        svd_features (np.ndarray): The structural SVD features corresponding to the dataframe.
        device (torch.device): The device to run inference on.
        batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE.
        tokenizer: Optional pre-loaded tokenizer. If None, loads from config.

    Returns:
        np.ndarray: Array of predicted probabilities (values between 0 and 1).
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    if tokenizer is None:
        tokenizer = get_tokenizer()

    # Create dataloader for inference
    # is_test=True ensures the dataset class does not look for labels
    dataloader = create_dataloader(
        df=df,
        svd_features=svd_features,
        tokenizer=tokenizer,
        batch_size=batch_size,
        is_train=False,
        is_test=True,
        shuffle=False,
    )

    # Run inference loop
    # model_inference returns concatenated predictions from the dataloader
    preds = model_inference(model, dataloader, device)

    return preds


def generate_pseudo_labels(
    test_df: pd.DataFrame,
    test_svd: np.ndarray,
    predictions: np.ndarray,
    high_thresh: Optional[float] = None,
    low_thresh: Optional[float] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generates a pseudo-labeled dataset by filtering test samples with high-confidence predictions.

    Args:
        test_df (pd.DataFrame): The test dataframe.
        test_svd (np.ndarray): SVD features for the test set.
        predictions (np.ndarray): Probability predictions for the test set.
        high_thresh (float, optional): Threshold above which a sample is labeled 1.
                                       Defaults to Config.PSEUDO_LABEL_HIGH.
        low_thresh (float, optional): Threshold below which a sample is labeled 0.
                                      Defaults to Config.PSEUDO_LABEL_LOW.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]:
            - pseudo_df: Dataframe containing the selected samples with an 'Insult' column.
            - pseudo_svd: Corresponding SVD features for the selected samples.
    """
    if high_thresh is None:
        high_thresh = Config.PSEUDO_LABEL_HIGH
    if low_thresh is None:
        low_thresh = Config.PSEUDO_LABEL_LOW

    # Ensure predictions are a 1D array for boolean indexing
    preds_flat = predictions.flatten()

    # Create boolean mask for high confidence samples
    # Select samples where probability is very high (Insult) or very low (Neutral)
    mask = (preds_flat >= high_thresh) | (preds_flat <= low_thresh)

    # Check if any samples met the criteria
    if not np.any(mask):
        print("No pseudo-labels generated (thresholds not met).")
        # Return empty structures with correct schema
        empty_df = pd.DataFrame(columns=list(test_df.columns) + ["Insult"])
        empty_svd = np.empty((0, test_svd.shape[1]))
        return empty_df, empty_svd

    # Filter Data and Features
    pseudo_df = test_df.iloc[mask].copy().reset_index(drop=True)
    pseudo_svd = test_svd[mask]

    # Assign Hard Labels
    # If prob >= 0.5 (which implies >= high_thresh), label is 1
    # If prob < 0.5 (which implies <= low_thresh), label is 0
    pseudo_labels = (preds_flat[mask] >= 0.5).astype(int)
    pseudo_df["Insult"] = pseudo_labels

    print(f"Generated {len(pseudo_df)} pseudo-labels from {len(test_df)} test samples.")

    return pseudo_df, pseudo_svd
