import os
import torch
import numpy as np
import random
import library.config as config
import library.data as data
import library.model as model_lib
import library.utils as utils


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(load_cached_data=False, epochs=None, debug_sample_size=None):
    """
    Main function to execute the training and inference pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data.
                                 Passed to the data loader factory.
        epochs (int, optional): Overrides the number of training epochs in config.
        debug_sample_size (int, optional): Overrides the dataset size for debugging.
    """

    # 1. Configuration Overrides
    if epochs is not None:
        config.EPOCHS = epochs
    if debug_sample_size is not None:
        config.DEBUG_SAMPLE_SIZE = debug_sample_size

    # 2. Reproducibility
    set_seed(config.SEED)

    print("Configuration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Debug Sample Size: {config.DEBUG_SAMPLE_SIZE}")

    # 3. Data Loading
    # The get_dataloaders function handles caching logic internally via the Sampler/Dataset
    print("\nInitializing DataLoaders...")
    train_loader, val_loader, test_loader = data.get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 4. Model Training
    # library.model.train_model handles:
    # - Model initialization
    # - Optimizer/Scheduler setup
    # - Training loop with CosineLoss
    # - Early stopping
    # - Saving the best model to config.MODEL_PATH
    print("\nStarting Model Training...")
    model = model_lib.train_model(train_loader, val_loader)

    # 5. Final Evaluation
    # We perform a final validation pass to print the metric with full precision
    print("\nPerforming Final Validation on Best Model...")
    device = config.DEVICE
    criterion = model_lib.CosineLoss()

    val_loss, val_metric = model_lib.validate(model, val_loader, criterion, device)

    print(f"Final Validation Loss: {val_loss}")
    print(f"Final Validation Metric (Mean Angular Error): {val_metric}")

    # 6. Submission Generation
    # Generates predictions for the test set and saves to config.SUBMISSION_PATH
    print("\nGenerating Submission...")
    model_lib.generate_submission(model, test_loader)

    print("\nPipeline Complete.")
