import os
import torch
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import train_whale_model, generate_predictions


def train_model(
    data_dir="./input",
    metadata_dir="./metadata",
    working_dir="./working/idea_1",
    submission_file="./submission/submission.csv",
    epochs=15,
    batch_size=32,
    lr=1e-4,
    patience=4,
    image_size=256,
    num_workers=4,
    load_cached_data=True,
    max_batches_per_epoch=None,
):
    """
    Orchestrates the training and prediction pipeline for Whale Species Classification.

    Args:
        data_dir (str): Root directory containing image data.
        metadata_dir (str): Directory containing train/val/test CSVs.
        working_dir (str): Directory to store checkpoints and cached data.
        submission_file (str): Path to save the final submission CSV.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        lr (float): Learning rate for the Adam optimizer.
        patience (int): Epochs to wait for improvement before early stopping.
        image_size (int): Target size for image resizing.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): If True, attempts to load LabelEncoder from cache.
        max_batches_per_epoch (int, optional): Limit batches per epoch for debugging.
    """

    # 1. Setup Environment
    set_seed(42)
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_file), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}")

    # 2. Prepare DataLoaders
    # This handles reading metadata, encoding labels (with caching), and creating DataLoaders
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader, label_encoder = get_dataloaders(
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
        cache_dir=working_dir,
        image_size=image_size,
    )

    num_classes = label_encoder.num_classes()
    print(f"Data preparation complete. Total classes: {num_classes}")

    # 3. Train Model
    # train_whale_model encapsulates the training loop, validation,
    # metric calculation (MAP@5), early stopping, and checkpointing.
    print(f"Starting training for up to {epochs} epochs...")
    model = train_whale_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        device=device,
        checkpoint_dir=working_dir,
        patience=patience,
        lr=lr,
        max_batches_per_epoch=max_batches_per_epoch,
    )

    # 4. Generate Submission
    # Moved to runfile.py to allow conditional execution based on validation score
    print("Training complete. Model saved.")
