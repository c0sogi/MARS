import time
import torch
import torch.nn as nn
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.neural_model import NeuralSolver


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio):
    """
    Runs one epoch of training.

    Args:
        model: The Seq2Seq model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Torch device.
        teacher_forcing_ratio: Probability of using teacher forcing.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        src = batch["input_ids"].to(device)
        trg = batch["target_ids"].to(device)

        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio)

        # Reshape for loss calculation
        # output: [batch_size, trg_len, output_dim] -> [(batch_size * trg_len) - 1, output_dim]
        # trg: [batch_size, trg_len] -> [(batch_size * trg_len) - 1]
        # We ignore the 0th index (SOS) for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The Seq2Seq model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["input_ids"].to(device)
            trg = batch["target_ids"].to(device)

            # Turn off teacher forcing for validation
            output = model(src, trg, 0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def run_training(debug_sample_size=None, load_cached_data=True):
    """
    Orchestrates the training process.

    Args:
        debug_sample_size (int, optional): Limit dataset size for debugging.
        load_cached_data (bool): Whether to load processed data from cache.
    """
    set_seed()

    # Load data
    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=load_cached_data,
        debug_sample_size=debug_sample_size,
    )

    # Initialize model using NeuralSolver wrapper to ensure consistent config usage
    # This sets up the Encoder, Decoder, Seq2Seq, Optimizer, and Criterion
    solver = NeuralSolver(tokenizer)
    model = solver.model
    optimizer = solver.optimizer
    criterion = solver.criterion
    device = solver.device

    print(f"Starting training on {device}...")
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            Config.TEACHER_FORCING_RATIO,
        )

        val_loss = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        # Print full precision as requested
        print(f"\tTrain Loss: {train_loss:.20f}")
        print(f"\t Val. Loss: {val_loss:.20f}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"\tSaved best model to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"\tNo improvement. Patience: {patience_counter}/{Config.PATIENCE}")
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break
