import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed
from library.model import Seq2SeqTransformer
from library.data_prep import prepare_curriculum_data


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5
):
    """
    Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and pi, after a warmup period during
    which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, clip_grad):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)

        # Shift targets for Seq2Seq:
        # Input to decoder: [BOS, t1, t2, ...] -> tgt[:, :-1]
        # Target for loss:  [t1, t2, ..., EOS] -> tgt[:, 1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        logits = model(src, tgt_input)

        # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len)
        output_dim = logits.shape[-1]
        loss = criterion(logits.reshape(-1, output_dim), tgt_output.reshape(-1))

        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_tokens += 1

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src_ids"].to(device)
            tgt = batch["tgt_ids"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)

            output_dim = logits.shape[-1]
            loss = criterion(logits.reshape(-1, output_dim), tgt_output.reshape(-1))
            total_loss += loss.item()

            # Calculate token-level accuracy (Teacher Forcing)
            # Ignore padding in accuracy calculation
            preds = torch.argmax(logits, dim=-1)
            mask = tgt_output != criterion.ignore_index

            correct = (preds == tgt_output) & mask
            correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


def train_model(load_cached_data=True):
    """
    Main training routine.

    Args:
        load_cached_data (bool): Whether to load pre-computed datasets/indices.

    Returns:
        model: The trained PyTorch model.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")

    # 1. Prepare Data
    train_dataset, val_dataset, _, char_tokenizer, target_tokenizer = (
        prepare_curriculum_data(load_cached_data=load_cached_data)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 2. Initialize Model
    src_vocab_size = char_tokenizer.get_vocab_size()
    tgt_vocab_size = target_tokenizer.get_vocab_size()

    # Pad indices
    src_pad_idx = char_tokenizer.char2id[char_tokenizer.pad_token]
    tgt_pad_idx = target_tokenizer.pad_token_id

    print(f"Initializing Model: Src Vocab={src_vocab_size}, Tgt Vocab={tgt_vocab_size}")

    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
    ).to(device)

    # 3. Setup Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate total steps for scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * Config.EPOCHS

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=Config.WARMUP_STEPS, num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_pad_idx, label_smoothing=Config.LABEL_SMOOTHING
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            Config.GRAD_CLIP,
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.10f}")
        print(f"  Val Loss:   {val_loss:.10f}")
        print(f"  Val Acc:    {val_acc:.10f}")

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(os.path.dirname(Config.BEST_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New best model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model before returning
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    return model
