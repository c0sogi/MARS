import os
import pandas as pd
import torch
from library.config import (
    SAMPLE_SUBMISSION_PATH,
    SUBMISSION_PATH,
    CACHE_DIR,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY,
    DEVICE,
    EMBED_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
)
from library.dataset import get_dataloaders
from library.model import BiLSTM, train_model, predict
from library.vocabulary import build_vocabulary
from library.utils import set_seed


def run_training_pipeline(
    load_cached_data: bool = True,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
    weight_decay: float = WEIGHT_DECAY,
    embed_dim: int = EMBED_DIM,
    hidden_dim: int = HIDDEN_DIM,
    output_dim: int = OUTPUT_DIM,
    device: str = DEVICE,
):
    """
    Orchestrates the training, validation, and prediction pipeline.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Early stopping patience.
        weight_decay (float): L2 regularization factor.
        embed_dim (int): Dimension of the embedding layer.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output layer.
        device (str): Compute device ('cpu' or 'cuda').
    """
    # Set random seeds for reproducibility
    set_seed()

    print("Initializing DataLoaders...")
    # get_dataloaders handles vocabulary building/loading and dataset creation
    # If load_cached_data is False, it will rebuild the vocab and cache it.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # Retrieve vocabulary size to initialize the model
    # We load the vocab that was just built/loaded by get_dataloaders
    # Since get_dataloaders runs first, the cache is guaranteed to be populated.
    vocab = build_vocabulary(load_cached_data=True)
    vocab_size = len(vocab.stoi)
    print(f"Vocabulary Size: {vocab_size}")

    print("Initializing Model...")
    model = BiLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    print("Starting Training...")
    # train_model handles the training loop, validation, early stopping, and saving the best model
    save_path = os.path.join(CACHE_DIR, "best_model.pth")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=learning_rate,
        patience=patience,
        weight_decay=weight_decay,
        save_path=save_path,
    )

    print("Generating Predictions on Test Set...")
    predictions = predict(trained_model, test_loader, device=device)

    print("Saving Submission...")
    if not os.path.exists(SAMPLE_SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Sample submission file not found at {SAMPLE_SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    # Basic validation of lengths
    if len(submission_df) != len(predictions):
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match sample submission rows ({len(submission_df)})."
        )

    # Assign predictions
    submission_df["Insult"] = predictions

    # Save to submission file
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {SUBMISSION_PATH}")
