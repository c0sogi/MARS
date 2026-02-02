import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, fbeta_score
from library.dataset import InkDataset
from library.model import SegFormerB2


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The SegFormer model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        device (torch.device): Compute device (CPU or CUDA).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move data to device
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (Loss is calculated inside the model forward method)
        outputs = model(images, labels)
        loss = outputs["loss"]

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The SegFormer model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Compute device.

    Returns:
        float: The F0.5 score on the validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(images)
            logits = outputs["logits"]

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Store predictions and targets on CPU to save GPU memory
            all_preds.append(probs.cpu())
            all_targets.append(labels.cpu())

    # Concatenate all batches
    if not all_preds:
        return 0.0

    predictions = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Calculate F0.5 Score
    score = fbeta_score(
        predictions,
        targets,
        beta=Config.F_BETA,
        threshold=Config.BINARIZATION_THRESHOLD,
    )

    return score


def run_training(num_epochs=None, batch_size=None, learning_rate=None, debug=False):
    """
    Main function to run the training and validation loop.

    Args:
        num_epochs (int, optional): Number of training epochs. Defaults to Config.
        batch_size (int, optional): Batch size. Defaults to Config.
        learning_rate (float, optional): Learning rate. Defaults to Config.
        debug (bool): If True, runs on a small subset of data for debugging.

    Returns:
        float: The best validation F0.5 score achieved.
    """
    # Set defaults from Config if not provided
    num_epochs = num_epochs if num_epochs is not None else Config.NUM_EPOCHS
    batch_size = batch_size if batch_size is not None else Config.BATCH_SIZE
    learning_rate = learning_rate if learning_rate is not None else Config.LEARNING_RATE

    # Ensure reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Datasets
    train_dataset = InkDataset(mode="train")
    val_dataset = InkDataset(mode="val")

    # Debug mode: slice datasets to verify pipeline
    if debug:
        print("Debug mode enabled: Using subset of data.")
        if len(train_dataset.df) > batch_size * 2:
            train_dataset.df = train_dataset.df.iloc[: batch_size * 2]
        if len(val_dataset.df) > batch_size:
            val_dataset.df = val_dataset.df.iloc[:batch_size]

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = SegFormerB2()
    model.to(device)

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # Training Loop
    best_score = Config.PREV_BEST_SCORE
    print(f"Starting training. Baseline Score to beat: {best_score}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_score = validate(model, val_loader, device)
        # Print full precision as requested
        print(f"Validation F0.5 Score: {val_score}")

        # Checkpoint Strategy
        if val_score > best_score:
            print(f"Score improved from {best_score} to {val_score}. Saving model...")
            best_score = val_score

            # Ensure working directory exists
            os.makedirs(Config.WORKING_DIR, exist_ok=True)
            save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
        else:
            print(f"Score did not improve (Best: {best_score}).")

    print(f"\nTraining completed. Best Validation Score: {best_score}")
    return best_score
