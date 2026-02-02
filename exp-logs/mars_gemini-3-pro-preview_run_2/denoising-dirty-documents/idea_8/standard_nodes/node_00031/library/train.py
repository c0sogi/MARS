import os
import torch
from torch.utils.data import DataLoader, Subset
from library.config import Config
from library.utils import seed_everything, get_logger, load_checkpoint
from library.dataset import DenoisingDataset
from library.model import CoConvNeXtUNet, train_model
from library.inference import generate_submission


def run_training(load_cached_data=True, num_epochs=None, batch_size=None, debug=False):
    """
    Orchestrates the training, validation, and submission generation process.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                 If False or cache missing, re-processes data.
        num_epochs (int, optional): Override the number of training epochs defined in Config.
        batch_size (int, optional): Override the training batch size defined in Config.
        debug (bool): If True, runs the pipeline on a small subset of data for debugging.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("TrainModule")

    logger.info(f"Initializing training pipeline on device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Apply Configuration Overrides
    if num_epochs is not None:
        Config.NUM_EPOCHS = num_epochs
        logger.info(f"Configuration update: NUM_EPOCHS set to {Config.NUM_EPOCHS}")

    if batch_size is not None:
        Config.BATCH_SIZE = batch_size
        logger.info(f"Configuration update: BATCH_SIZE set to {Config.BATCH_SIZE}")

    # 3. Prepare Datasets and Loaders
    logger.info("Loading training and validation datasets...")

    train_dataset = DenoisingDataset(
        Config.TRAIN_CSV, mode="train", load_cached_data=load_cached_data
    )
    val_dataset = DenoisingDataset(
        Config.VAL_CSV, mode="val", load_cached_data=load_cached_data
    )

    if debug:
        logger.info("Debug mode enabled: using data subsets.")
        # Subset training data (limit patches)
        train_indices = range(min(len(train_dataset), 10 * Config.PATCHES_PER_IMAGE))
        train_dataset = Subset(train_dataset, train_indices)

        # Subset validation data (limit images)
        val_indices = range(min(len(val_dataset), 5))
        val_dataset = Subset(val_dataset, val_indices)

    # Train loader: Shuffled, batched
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val loader: Batch size 1 because validation images have varying dimensions
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model
    logger.info("Building CoConvNeXtUNet model...")
    model = CoConvNeXtUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_filters=Config.BASE_FILTERS,
    ).to(device)

    # 5. Execute Training Loop
    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, device)

    # 6. Retrieve and Print Best Metric
    # Load the best checkpoint to ensure we have the optimal weights and to print the exact metric
    try:
        _, best_rmse = load_checkpoint(Config.MODEL_PATH, model, device=device)
        logger.info(f"Training finished. Best Validation RMSE: {best_rmse}")
    except Exception as e:
        logger.error(f"Failed to load best model checkpoint: {e}")

    # 7. Inference and Submission
    logger.info("Starting inference generation...")

    # Free up memory before inference
    del train_loader, val_loader, train_dataset, val_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load Test Data
    test_dataset = DenoisingDataset(
        Config.TEST_CSV, mode="test", load_cached_data=load_cached_data
    )

    if debug:
        test_dataset = Subset(test_dataset, range(min(len(test_dataset), 5)))

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Test images vary in size
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Generate Submission
    generate_submission(test_loader, device)

    logger.info("Pipeline completed successfully.")
