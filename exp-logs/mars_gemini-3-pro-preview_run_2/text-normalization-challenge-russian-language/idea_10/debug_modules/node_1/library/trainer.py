import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library import config
from library import utils
from library import model as model_lib


def train_epoch(
    model, dataloader, optimizer, criterion, scheduler, device, src_pad_id, tgt_pad_id
):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        src = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Prepare Decoder Inputs and Targets
        # labels: [batch, seq_len] containing [SOS, ..., EOS, PAD]
        # tgt_input: [SOS, ..., token_n] (exclude last)
        # tgt_out:   [token_1, ..., EOS] (exclude first)
        tgt_input = labels[:, :-1]
        tgt_out = labels[:, 1:]

        # Generate Masks
        # Padding masks: True where value is pad_id
        src_key_padding_mask = src == src_pad_id
        tgt_key_padding_mask = tgt_input == tgt_pad_id

        # Causal mask for decoder
        tgt_seq_len = tgt_input.size(1)
        tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward Pass
        # output shape: [batch, tgt_seq_len, tgt_vocab_size]
        output = model(
            src=src,
            tgt=tgt_input,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )

        # Compute Loss
        # Flatten output and targets
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_out.reshape(-1))

        # Backward Pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

        # Optimizer Step
        optimizer.step()

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, src_pad_id, tgt_pad_id):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            tgt_input = labels[:, :-1]
            tgt_out = labels[:, 1:]

            src_key_padding_mask = src == src_pad_id
            tgt_key_padding_mask = tgt_input == tgt_pad_id
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(
                device
            )

            output = model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask,
            )

            loss = criterion(output.reshape(-1, output.size(-1)), tgt_out.reshape(-1))

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(train_dataset, val_dataset, char_tokenizer, bpe_tokenizer):
    """
    Main function to setup model, optimizer, and run the training loop.
    """
    utils.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Trainer: Using device {device}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # Initialize Model
    print("Trainer: Initializing SemioticTransformer...")
    model = model_lib.SemioticTransformer(
        src_vocab_size=char_tokenizer.vocab_size,
        tgt_vocab_size=bpe_tokenizer.vocab_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
    ).to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Criterion
    # Ignore the padding index in the target
    criterion = nn.CrossEntropyLoss(
        ignore_index=bpe_tokenizer.pad_id, label_smoothing=config.LABEL_SMOOTHING
    )

    # Scheduler
    # OneCycleLR handles warmup and cosine decay
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=config.WARMUP_STEPS / total_steps if total_steps > 0 else 0.1,
        anneal_strategy="cos",
    )

    # Training State
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Trainer: Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
            src_pad_id=char_tokenizer.pad_id,
            tgt_pad_id=bpe_tokenizer.pad_id,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            src_pad_id=char_tokenizer.pad_id,
            tgt_pad_id=bpe_tokenizer.pad_id,
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch}/{config.NUM_EPOCHS} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f}"
        )

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  -> New best model saved to {config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(
                f"  -> Validation loss did not improve. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Trainer: Early stopping triggered.")
            break

    print("Trainer: Training complete.")
    return model
