import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed
from library.transformer_data import prepare_dataloaders
from library.transformer_model import Seq2SeqTransformer


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        # Prepare inputs and targets for Teacher Forcing
        # Decoder Input: <SOS> ... x_n
        tgt_input = tgt[:, :-1]

        # Target Label: x_1 ... <EOS>
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt_input)

        # Reshape for Loss: [batch_size * seq_len, vocab_size]
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        tgt_out = tgt_out.reshape(-1)

        loss = criterion(output, tgt_out)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            output = model(src, tgt_input)

            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            tgt_out = tgt_out.reshape(-1)

            loss = criterion(output, tgt_out)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def fit_transformer(load_cached_data=True, debug=False):
    """
    Main function to train the Transformer model.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # 1. Prepare Data
    train_loader, val_loader, tokenizer = prepare_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    vocab_size = len(tokenizer)
    print(f"Vocabulary Size: {vocab_size}")

    # 2. Initialize Model
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    # 3. Setup Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Ignore padding token in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Scheduler: OneCycleLR is efficient for convergence
    # We estimate total steps
    total_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.1,  # Warmup for first 10%
        anneal_strategy="cos",
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\tTrain Loss: {train_loss}")
        print(f"\t Val. Loss: {val_loss}")

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            print(
                f"\tValidation loss decreased. Saving model to {Config.MODEL_CHECKPOINT}"
            )
        else:
            patience_counter += 1
            print(
                f"\tValidation loss did not decrease. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete.")
