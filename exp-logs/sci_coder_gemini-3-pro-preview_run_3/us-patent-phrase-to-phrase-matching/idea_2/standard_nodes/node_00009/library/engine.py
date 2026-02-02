import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import CrossEncoderModel, train_model, generate_submission


def train_fn(
    debug=Config.debug,
    epochs=Config.epochs,
    batch_size=Config.train_batch_size,
    learning_rate=Config.learning_rate,
    weight_decay=Config.weight_decay,
    warmup_ratio=Config.warmup_ratio,
    patience=Config.patience,
    save_path=Config.model_save_path,
    submission_path=Config.submission_path,
):
    """
    Main function to setup components and run the training and inference pipeline.

    Args:
        debug (bool): If True, runs on a small subset of data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        warmup_ratio (float): Ratio of total steps to use for warmup.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model checkpoint.
        submission_path (str): Path to save the submission CSV.
    """
    # 1. Setup Environment
    seed_everything(Config.seed)
    device = Config.device

    # Update Config with runtime arguments to ensure consistency
    Config.debug = debug
    Config.epochs = epochs
    Config.train_batch_size = batch_size
    Config.learning_rate = learning_rate
    Config.weight_decay = weight_decay
    Config.warmup_ratio = warmup_ratio
    Config.patience = patience
    Config.model_save_path = save_path
    Config.submission_path = submission_path

    print(f"Initializing Engine on Device: {device}")

    # 2. Prepare Data
    # Initialize tokenizer associated with the backbone
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Get DataLoaders (caching is handled internally by dataset.py)
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer, load_cached_data=True, debug=debug
    )

    # 3. Initialize Model
    model = CrossEncoderModel(
        model_name=Config.model_name,
        num_labels=Config.num_labels,
        dropout=Config.dropout,
    )

    # 4. Optimizer & Scheduler
    # Using torch.optim.AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Calculate training steps for the scheduler
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Run Training
    # train_model handles the training loop, validation, early stopping, and saving
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        patience=patience,
        save_path=save_path,
    )

    # 6. Inference
    # generate_submission handles prediction on test set and file saving
    generate_submission(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        submission_path=submission_path,
    )

    return trained_model
