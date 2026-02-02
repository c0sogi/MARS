import os
import torch
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_processing import create_dataloaders
from library.model import ChatbotMLP
from library.trainer import ModelTrainer

# Initialize logger for this module
logger = get_logger("inference")


def generate_submission(
    load_cached_data: bool = True,
    batch_size: int = Config.BATCH_SIZE,
    debug: bool = Config.DEBUG,
):
    """
    Orchestrates the inference process for the Chatbot Preference Prediction task.

    Steps:
    1. Sets the random seed and debug configuration.
    2. Loads the necessary DataLoaders (specifically the test loader).
    3. Initializes the ChatbotMLP model with the correct input dimension.
    4. Initializes the ModelTrainer to handle checkpoint loading and inference.
    5. Generates predictions and saves the submission file.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed embeddings from disk.
                                 If False or file missing, re-computes embeddings.
        batch_size (int): The batch size to use for the DataLoader.
        debug (bool): If True, runs the pipeline on a small subset of the data.
    """
    # Update Config based on arguments
    Config.DEBUG = debug

    # Ensure reproducibility
    seed_everything(Config.SEED)

    logger.info("Initializing inference pipeline...")

    # 1. Prepare DataLoaders
    # We call create_dataloaders to ensure the test set is processed and loaded.
    # While train/val loaders are returned, they are not primarily used for inference
    # but are required by the ModelTrainer signature.
    logger.info(f"Loading data (Cached: {load_cached_data}, Debug: {debug})...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    # The input dimension is determined by the feature construction in ChatbotDataset.
    # Features are concatenated: [Prompt, ResA, ResB, Diff, Prod]
    # Each component has size Config.EMBEDDING_DIM.
    # Total Input Dim = 5 * Config.EMBEDDING_DIM
    input_dim = Config.EMBEDDING_DIM * 5

    logger.info(f"Initializing ChatbotMLP with input dimension: {input_dim}")
    model = ChatbotMLP(
        input_dim=input_dim,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
        num_classes=Config.NUM_CLASSES,
    )

    # 3. Initialize Trainer
    # The ModelTrainer encapsulates the logic for loading the best checkpoint
    # and running the inference loop.
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # 4. Generate Submission
    # This method loads the weights from Config.MODEL_SAVE_PATH, predicts on test_loader,
    # and saves the results to Config.SUBMISSION_PATH.
    logger.info("Starting prediction generation...")
    trainer.generate_submission()

    logger.info("Inference pipeline completed.")
