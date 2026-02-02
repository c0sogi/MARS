import os
import torch
import numpy as np
import pandas as pd
from library import config, model, data, utils


def inference_fn(model_path, device, load_cached_data=True):
    """
    Loads the trained model and generates predictions for the test dataset.

    Args:
        model_path (str): Path to the saved model state dictionary.
        device (torch.device): The compute device (CPU or GPU).
        load_cached_data (bool): Whether to use cached metadata/dataloaders.

    Returns:
        np.ndarray: A 1D numpy array containing the predicted probabilities for the test set.
    """
    # Initialize the model architecture
    net = model.MetadataEfficientNet()
    net.to(device)

    # Load model weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)
    else:
        print(f"Warning: Model file {model_path} not found. Using initialized weights.")

    # Set model to evaluation mode
    # This automatically disables StochasticModalityDropout and BatchNorm updates
    net.eval()

    # Get the test dataloader
    # We only need the test_loader (index 2)
    _, _, test_loader = data.get_dataloaders(load_cached_data=load_cached_data)

    all_probs = []

    # Run inference
    with torch.no_grad():
        for images, ages, implants in test_loader:
            images = images.to(device)
            ages = ages.to(device)
            implants = implants.to(device)

            # Forward pass
            # Inputs: (B, 1, H, W), (B,), (B,)
            logits = net(images, ages, implants)

            # Apply sigmoid to get probabilities in range [0, 1]
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches into a single array
    if len(all_probs) > 0:
        flat_probs = np.concatenate(all_probs).flatten()
    else:
        flat_probs = np.array([])

    return flat_probs


def create_submission(probs, output_path, load_cached_data=True):
    """
    Aggregates image-level predictions to prediction-level (breast-level) scores
    and generates the submission CSV.

    Args:
        probs (np.ndarray): Predicted probabilities corresponding to the test metadata order.
        output_path (str): File path to save the submission CSV.
        load_cached_data (bool): Whether to load the test metadata from cache.
    """
    # Load the test metadata to map predictions to prediction_ids
    # We prioritize the parquet cache to ensure alignment with the dataloader
    cache_path = os.path.join(config.WORKING_DIR, "processed_test.parquet")

    if load_cached_data and os.path.exists(cache_path):
        df_test = pd.read_parquet(cache_path)
    else:
        # If cache is missing or forced reload, process from scratch
        # This ensures we have the dataframe even if the cache was deleted
        _, _, df_test = data.process_metadata(load_cached_data=False)

    # Safety check for length alignment
    if len(probs) != len(df_test):
        print(
            f"Warning: Number of predictions ({len(probs)}) does not match metadata length ({len(df_test)})."
        )
        # Align lengths to prevent errors, assuming sequential truncation is safest fallback
        min_len = min(len(probs), len(df_test))
        df_test = df_test.iloc[:min_len]
        probs = probs[:min_len]

    # Assign image-level probabilities
    df_test["cancer"] = probs

    # Aggregate predictions by prediction_id
    # Strategy: Max probability across all views (images) for a specific breast
    submission_df = df_test.groupby("prediction_id")["cancer"].max().reset_index()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the submission file
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_prediction(
    model_path=config.MODEL_SAVE_PATH,
    output_path=config.SUBMISSION_PATH,
    load_cached_data=True,
):
    """
    Main execution function for the prediction module.

    Args:
        model_path (str): Path to the trained model.
        output_path (str): Path to save the submission CSV.
        load_cached_data (bool): Whether to use cached data.
    """
    utils.seed_everything(config.SEED)
    device = config.DEVICE

    print(f"Starting inference with model: {model_path}")

    # 1. Generate Probabilities
    probs = inference_fn(model_path, device, load_cached_data=load_cached_data)

    # 2. Create Submission File
    create_submission(probs, output_path, load_cached_data=load_cached_data)
