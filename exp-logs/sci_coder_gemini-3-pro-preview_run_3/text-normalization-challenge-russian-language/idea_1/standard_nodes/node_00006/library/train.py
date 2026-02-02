import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os
from library.config import Config, set_seed
from library.vocabulary import CharVocab
from library.dataset import get_dataloader
from library.model import Encoder, Decoder, Seq2Seq, Attention
from library.utils import count_parameters, save_checkpoint


def train_epoch(model, iterator, optimizer, criterion, clip, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The Seq2Seq model.
        iterator (DataLoader): Training data loader.
        optimizer (optim.Optimizer): Optimizer instance.
        criterion (nn.Module): Loss function.
        clip (float): Gradient clipping value.
        device (torch.device): Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    epoch_loss = 0
    batch_count = 0

    for i, batch in enumerate(iterator):
        if batch is None:
            continue

        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # teacher_forcing_ratio is handled inside the model.forward via Config or default
        output = model(src, tgt, teacher_forcing_ratio=Config.teacher_forcing_ratio)

        # output shape: [batch size, trg len, output dim]
        # tgt shape: [batch size, trg len]

        output_dim = output.shape[-1]

        # Discard the first token (SOS) from output and target for loss calculation
        # output becomes [(batch size * trg len - 1), output dim]
        # tgt becomes [(batch size * trg len - 1)]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)

        # Calculate loss
        loss = criterion(output, tgt)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    return epoch_loss / batch_count if batch_count > 0 else 0


def evaluate(model, iterator, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The Seq2Seq model.
        iterator (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Average loss for the epoch.
    """
    model.eval()
    epoch_loss = 0
    batch_count = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if batch is None:
                continue

            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            # Turn off teacher forcing for evaluation to test generation capability
            output = model(src, tgt, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]

            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)
            epoch_loss += loss.item()
            batch_count += 1

    return epoch_loss / batch_count if batch_count > 0 else 0


def train_model(
    num_epochs=Config.num_epochs,
    batch_size=Config.batch_size,
    load_cached_data=True,
    learning_rate=Config.learning_rate,
):
    """
    Main function to train the model.

    Args:
        num_epochs (int): Number of epochs to train.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to load cached data.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        model (nn.Module): The trained model.
    """
    # 1. Setup
    set_seed(42)
    device = Config.device
    print(f"Using device: {device}")

    # 2. Data Preparation
    vocab = CharVocab()
    # Build or load vocabulary
    vocab.build_vocab(Config.TRAIN_DATA_PATH, load_cached_data=load_cached_data)

    print(f"Vocabulary size: {len(vocab)}")

    train_loader = get_dataloader(
        Config.TRAIN_DATA_PATH,
        vocab,
        batch_size,
        is_test=False,
        shuffle=True,
        load_cached_data=load_cached_data,
    )

    val_loader = get_dataloader(
        Config.VAL_DATA_PATH,
        vocab,
        batch_size,
        is_test=False,
        shuffle=False,
        load_cached_data=load_cached_data,
    )

    # 3. Model Initialization
    enc = Encoder(
        input_dim=len(vocab),
        emb_dim=Config.embed_dim,
        hid_dim=Config.hidden_dim,
        n_layers=Config.n_layers,
        dropout=Config.dropout,
    )

    attn = Attention(Config.hidden_dim, Config.hidden_dim)
    dec = Decoder(
        output_dim=len(vocab),
        emb_dim=Config.embed_dim,
        enc_hid_dim=Config.hidden_dim,
        dec_hid_dim=Config.hidden_dim,
        n_layers=Config.n_layers,
        dropout=Config.dropout,
        attention=attn,
    )

    model = Seq2Seq(enc, dec, device).to(device)

    # Initialize weights
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)

    print(f"The model has {count_parameters(model):,} trainable parameters")

    # 4. Optimization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    best_valid_loss = float("inf")
    patience_counter = 0

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, Config.clip_grad, device
        )
        valid_loss = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # Print full precision as requested
        print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\tTrain Loss: {train_loss}")
        print(f"\t Val. Loss: {valid_loss}")

        # Checkpoint and Early Stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, optimizer, epoch, valid_loss, Config.CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return model
