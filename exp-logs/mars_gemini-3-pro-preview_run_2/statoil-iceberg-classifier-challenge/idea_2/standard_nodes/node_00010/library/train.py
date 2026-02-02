import torch
import torch.nn as nn
import torch.optim as optim
from library.config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    DROPOUT_RATE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    DEBUG,
    DEBUG_SIZE,
)
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import SAHCN, train_model, predict_and_submit


def run_training_pipeline(
    load_cached_data=True,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    dropout_rate=DROPOUT_RATE,
    debug=DEBUG,
    debug_size=DEBUG_SIZE,
    model_save_path=MODEL_SAVE_PATH,
    submission_path=SUBMISSION_PATH,
):
    """
    Orchestrates the training and submission pipeline for the Iceberg Classifier.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from disk cache.
        batch_size (int): Batch size for training and inference.
        learning_rate (float): Learning rate for the Adam optimizer.
        num_epochs (int): Maximum number of training epochs.
        patience (int): Patience for early stopping.
        dropout_rate (float): Dropout rate for the model's dense layers.
        debug (bool): If True, runs on a small subset of data.
        debug_size (int): Number of samples to use in debug mode.
        model_save_path (str): Path to save the best model checkpoint.
        submission_path (str): Path to save the submission CSV.
    """
    # 1. Set Random Seeds for reproducibility
    seed_everything()

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load Data
    # get_dataloaders handles caching, metadata splitting, and preprocessing
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data,
        batch_size=batch_size,
        debug=debug,
        debug_size=debug_size,
    )

    # 4. Initialize Model
    # SAHCN: Spatially-Aware Hybrid Convolutional Network
    model = SAHCN(dropout_rate=dropout_rate)

    # 5. Define Loss and Optimizer
    # We use BCEWithLogitsLoss as the model outputs raw logits
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Scheduler (Cite solution_lesson_node_00009)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 6. Train Model
    # Executes the training loop with validation monitoring and early stopping.
    # Returns the model with the best validation loss loaded.
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        patience=patience,
        device=device,
        scheduler=scheduler,
        save_path=model_save_path,
    )

    # 7. Generate Submission
    # Predicts on the test set and saves the results to a CSV file.
    predict_and_submit(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        submission_path=submission_path,
    )
