import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.data_loader import get_data, GNSSDataset
from library.model import WindowedMLP, train_model, generate_submission


def run_experiment(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.EARLY_STOPPING_PATIENCE,
    load_cached_data=True,
):
    """
    Orchestrates the entire training and inference pipeline.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        patience (int): Patience for early stopping.
        load_cached_data (bool): Whether to load processed data from cache if available.
    """

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("Step 1: Loading Training and Validation Data...")

    # Load training data
    # get_data returns (X, y, meta_list) for train/val splits
    X_train, y_train, _ = get_data(split="train", load_cached_data=load_cached_data)

    # Load validation data
    X_val, y_val, _ = get_data(split="val", load_cached_data=load_cached_data)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # -------------------------------------------------------------------------
    # 2. Dataset and DataLoader Creation
    # -------------------------------------------------------------------------
    print("Step 2: Creating DataLoaders...")

    train_dataset = GNSSDataset(X_train, y_train, mode="train")
    val_dataset = GNSSDataset(X_val, y_val, mode="val")

    # Use pin_memory=True if using CUDA for faster data transfer
    use_pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Step 3: Initializing Model...")

    model = WindowedMLP(
        input_dim=Config.INPUT_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        output_dim=Config.OUTPUT_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Step 4: Starting Training...")

    # train_model handles the training loop, validation, metrics, optimization, and checkpointing
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        device=Config.DEVICE,
        checkpoint_path=Config.MODEL_CHECKPOINT_PATH,
    )

    print("Training completed.")
    return trained_model
