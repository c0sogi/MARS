import os
import torch
import pandas as pd
from library.config import Config
from library.data_processing import get_dataloaders
from library.model import LayerNormFunnelMLP, train_model, predict


def run_training(load_cached_data=True, epochs=None, batch_size=None):
    """
    Manages the training lifecycle for the Layer-Normalized Funnel MLP.

    This function orchestrates the pipeline by:
    1. Setting up the environment and configuration.
    2. Loading (and optionally processing) the dataset.
    3. Initializing the model architecture.
    4. executing the training loop with early stopping.
    5. Generating and saving predictions for the test set.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
                                 If False or cache missing, re-processes data.
        epochs (int, optional): Override for the number of training epochs.
        batch_size (int, optional): Override for the batch size.

    Returns:
        model (nn.Module): The trained PyTorch model with the best weights loaded.
    """
    # 1. Setup Configuration and Seeds
    Config.setup()

    # Apply overrides or defaults
    if epochs is None:
        epochs = Config.EPOCHS
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    print(
        f"Starting training run. Epochs: {epochs}, Batch Size: {batch_size}, Device: {Config.DEVICE}"
    )

    # 2. Data Loading
    # get_dataloaders handles caching logic internally via process_and_cache_data
    train_loader, val_loader, test_loader, vocab_sizes = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    # Determine the number of continuous features from the config definition
    cont_dim = len(Config.CONTINUOUS_COLS)

    print(
        f"Initializing LayerNormFunnelMLP with {len(vocab_sizes)} categorical features and {cont_dim} continuous features."
    )

    model = LayerNormFunnelMLP(
        vocab_sizes=vocab_sizes,
        cont_dim=cont_dim,
        embed_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        token_dropout_rate=Config.TOKEN_DROPOUT_RATE,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # 4. Training
    # train_model handles optimizer (AdamW), scheduler (OneCycleLR), loop, validation, and checkpointing.
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=Config.LEARNING_RATE,
        device=Config.DEVICE,
    )

    # 5. Inference
    print("Generating predictions on the test set...")
    submission_df = predict(model, test_loader, device=Config.DEVICE)

    # 6. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

    return model
