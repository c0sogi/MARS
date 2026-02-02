import os
import torch
from transformers import get_linear_schedule_with_warmup
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import get_model


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Performs one training epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: The optimizer (AdamW).
        scheduler: Learning rate scheduler.
        device: Torch device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        # Backward pass
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_loss(model, dataloader, device):
    """
    Computes validation loss.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        device: Torch device.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_runner(seed, debug=False):
    """
    Orchestrates the full training process for a specific seed.

    Args:
        seed (int): Random seed for initialization.
        debug (bool): Whether to run in debug mode (fewer data).
    """
    set_seed(seed)

    # Load Data
    # get_dataloaders handles caching internally
    train_loader, val_loader, _ = get_dataloaders(debug=debug, load_cached_data=True)

    device = Config.DEVICE
    model = get_model()

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = Config.EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    save_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pt")

    print(f"Starting training for seed {seed}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate_loss(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Clean up to save memory
    del model, optimizer, scheduler
    torch.cuda.empty_cache()


def run_training_pipeline(debug=False):
    """
    Runs training for all seeds defined in Config.
    """
    for seed in Config.SEEDS:
        train_runner(seed, debug=debug)
