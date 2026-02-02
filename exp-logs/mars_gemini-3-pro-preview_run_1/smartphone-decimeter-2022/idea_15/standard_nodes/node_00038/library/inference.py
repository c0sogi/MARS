import os
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import get_logger
from library.data_processing import get_data
from library.dataset import get_datasets
from library.model import AttentionGatedResUNet1D
from library.trainer import generate_submission


def run_inference(
    load_cached_data: bool = True,
    batch_size: int = Config.BATCH_SIZE,
    device: str = Config.DEVICE,
):
    """
    Runs the inference pipeline: loads data, prepares datasets, loads model,
    and generates the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        batch_size (int): Batch size for inference.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """
    logger = get_logger()
    logger.info("Starting Inference Pipeline...")

    # 1. Load Data
    # We need train_df to ensure the test dataset is normalized using training statistics
    # get_data handles the caching logic internally
    logger.info("Loading data (Train/Val/Test)...")
    train_df, val_df, test_df = get_data(load_cached_data=load_cached_data)

    if test_df.empty:
        logger.error("Test data is empty. Cannot proceed with inference.")
        return

    # 2. Prepare Datasets
    # get_datasets calculates mean/std from train_df and applies it to test_dataset
    logger.info("Preparing datasets and normalizing...")
    # We ignore the returned train/val datasets as we only need test for inference
    _, _, test_dataset = get_datasets(train_df, val_df, test_df)

    # 3. Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model
    logger.info(f"Initializing model on {device}...")
    model = AttentionGatedResUNet1D().to(device)

    # 5. Load Trained Weights
    model_path = Config.MODEL_SAVE_PATH
    if os.path.exists(model_path):
        logger.info(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logger.error(
            f"Model checkpoint not found at {model_path}. Please train the model first."
        )
        return

    # 6. Generate Submission
    # This function handles the forward pass, coordinate conversion (ENU -> LatLon),
    # and saving the results to the submission CSV file defined in Config.
    logger.info("Generating submission file...")
    generate_submission(model, test_loader, device)

    logger.info("Inference Pipeline Completed.")
