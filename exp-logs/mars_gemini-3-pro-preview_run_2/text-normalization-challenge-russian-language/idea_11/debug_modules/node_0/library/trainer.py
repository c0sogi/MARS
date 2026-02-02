import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset

from library.config import Config
from library.utils import get_device, set_seed, print_metrics, cleanup, load_raw_data
from library.transformer_arch import CharToSubwordTransformer
from library.data_manager import get_dataloaders, _add_context_columns
from library.hfbb_layer import HFBBModel
from library.tokenization import CharTokenizer, BPETokenizer


def train_model(
    char_tokenizer: CharTokenizer,
    bpe_tokenizer: BPETokenizer,
    num_epochs: int = Config.NUM_EPOCHS,
    debug_subset_size: int = Config.DEBUG_SUBSET_SIZE,
):
    """
    Orchestrates the training of the CharToSubwordTransformer.
    """
    set_seed(Config.SEED)
    device = get_device()

    # Patch Config for debug size if necessary
    original_subset_size = Config.DEBUG_SUBSET_SIZE
    Config.DEBUG_SUBSET_SIZE = debug_subset_size

    print(f"Initializing training on {device}...")

    # 1. Data Preparation
    train_loader, val_loader = get_dataloaders(char_tokenizer, bpe_tokenizer)

    # 2. Model Initialization
    src_vocab_size = len(char_tokenizer)
    tgt_vocab_size = len(bpe_tokenizer)
    src_pad_idx = char_tokenizer.pad_token_id
    tgt_pad_idx = bpe_tokenizer.pad_token_id

    model = CharToSubwordTransformer(
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
        max_src_len=Config.MAX_SRC_LEN,
        max_tgt_len=Config.MAX_TGT_LEN,
    ).to(device)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=Config.WARMUP_STEPS, num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_pad_idx, label_smoothing=Config.LABEL_SMOOTHING
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs.")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

            # Teacher forcing:
            # Input to decoder: [BOS, t1, t2, ..., tn]
            # Target for loss:  [t1, t2, ..., tn, EOS]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()

            logits = model(src, tgt_input)  # [Batch, SeqLen, Vocab]

            # Flatten for loss calculation
            loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = model(src, tgt_input)
                loss = criterion(
                    logits.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1)
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time

        # Metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "time": epoch_time,
        }
        print_metrics(metrics)

        # 6. Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

        cleanup()

    # Restore Config
    Config.DEBUG_SUBSET_SIZE = original_subset_size

    # Load best model before returning
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    return model


def greedy_decode(model, src, max_len, start_symbol, end_symbol, device):
    """
    Performs greedy decoding for a batch of source sequences.
    """
    src = src.to(device)
    batch_size = src.size(0)

    # Encode
    memory = model.encode(src)

    # Initialize decoder input with SOS
    ys = torch.full((batch_size, 1), start_symbol, dtype=torch.long).to(device)

    # Track finished sequences
    finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

    for _ in range(max_len - 1):
        # Decode
        out = model.decode(ys, memory)
        # Project to vocab
        prob = model.generator(out[:, -1])
        # Greedy choice
        _, next_word = torch.max(prob, dim=1)

        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)

        # Update finished status
        finished |= next_word.squeeze() == end_symbol

        if finished.all():
            break

    return ys


def generate_submission(char_tokenizer: CharTokenizer, bpe_tokenizer: BPETokenizer):
    """
    Generates the submission file using the hybrid HFBB + Transformer pipeline.
    """
    print("Starting submission generation...")
    device = get_device()

    # 1. Load Test Data
    df_test = load_raw_data("test")
    print(f"Test data loaded: {len(df_test)} rows.")

    # Add context columns
    df_test = _add_context_columns(df_test)

    # 2. Initialize HFBB (Tier 1)
    hfbb = HFBBModel(load_cached_data=True)

    # 3. Initialize Transformer (Tier 2)
    src_vocab_size = len(char_tokenizer)
    tgt_vocab_size = len(bpe_tokenizer)

    model = CharToSubwordTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_idx=char_tokenizer.pad_token_id,
        tgt_pad_idx=bpe_tokenizer.pad_token_id,
    ).to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print(
            "Warning: Best model checkpoint not found. Using initialized weights (expect poor performance)."
        )

    model.eval()

    # 4. Inference Pipeline
    predictions = []

    # We will process in chunks to manage memory, but for the transformer part we need batching.
    # First, run HFBB on everything.
    print("Running Tier 1: HFBB Inference...")

    # Pre-calculate masks for efficiency
    semiotic_mask = (
        df_test["before"].astype(str).str.contains(Config.SEMIOTIC_REGEX, regex=True)
    )

    results = [None] * len(df_test)
    transformer_indices = []

    # Iterate to apply HFBB
    # Converting to list for speed
    befores = df_test["before"].astype(str).tolist()
    prev1s = df_test["prev_1"].fillna("").astype(str).tolist()
    next1s = df_test["next_1"].fillna("").astype(str).tolist()

    for i in range(len(df_test)):
        # Step 1 & 2: HFBB Query
        pred = hfbb.query(befores[i], prev1s[i], next1s[i])

        if pred is not None:
            results[i] = pred
        else:
            # Step 3: Check fallback
            if semiotic_mask[i]:
                # Needs Transformer
                transformer_indices.append(i)
            else:
                # Identity Fallback
                results[i] = befores[i]

    print(f"HFBB resolved {len(df_test) - len(transformer_indices)} tokens.")
    print(f"Sending {len(transformer_indices)} tokens to Tier 2: Transformer.")

    # 5. Run Transformer on remaining indices
    if transformer_indices:
        # Prepare data for transformer
        # We need to construct source sequences similar to SemioticDataset

        batch_size = Config.BATCH_SIZE
        num_batches = (len(transformer_indices) + batch_size - 1) // batch_size

        sep_id = char_tokenizer.sep_token_id

        # Helper to encode
        def encode_text(text):
            return char_tokenizer.encode(text, add_special_tokens=False)

        space_ids = encode_text(" ")

        for b in range(num_batches):
            batch_idxs = transformer_indices[b * batch_size : (b + 1) * batch_size]

            src_batch = []

            for idx in batch_idxs:
                # Construct context
                p2 = str(df_test.iloc[idx].get("prev_2", ""))
                p1 = str(df_test.iloc[idx].get("prev_1", ""))
                curr = str(df_test.iloc[idx]["before"])
                n1 = str(df_test.iloc[idx].get("next_1", ""))
                n2 = str(df_test.iloc[idx].get("next_2", ""))

                src_ids = []
                if p2:
                    src_ids.extend(encode_text(p2) + space_ids)
                if p1:
                    src_ids.extend(encode_text(p1))
                src_ids.append(sep_id)
                src_ids.extend(encode_text(curr))
                src_ids.append(sep_id)
                if n1:
                    src_ids.extend(encode_text(n1) + space_ids)
                if n2:
                    src_ids.extend(encode_text(n2))

                if len(src_ids) > Config.MAX_SRC_LEN:
                    src_ids = src_ids[: Config.MAX_SRC_LEN]

                src_batch.append(torch.tensor(src_ids, dtype=torch.long))

            # Pad
            src_padded = torch.nn.utils.rnn.pad_sequence(
                src_batch, batch_first=True, padding_value=char_tokenizer.pad_token_id
            )

            # Inference
            with torch.no_grad():
                generated_ids = greedy_decode(
                    model,
                    src_padded,
                    max_len=Config.MAX_TGT_LEN,
                    start_symbol=bpe_tokenizer.sos_token_id,
                    end_symbol=bpe_tokenizer.eos_token_id,
                    device=device,
                )

            # Decode BPE
            for k, g_ids in enumerate(generated_ids):
                # Convert tensor to list
                ids_list = g_ids.cpu().tolist()
                decoded_text = bpe_tokenizer.decode(ids_list, skip_special_tokens=True)

                # Store result
                original_idx = batch_idxs[k]
                results[original_idx] = decoded_text

            if (b + 1) % 100 == 0:
                print(f"Processed {b + 1}/{num_batches} transformer batches.")
                cleanup()

    # 6. Save Submission
    print("Saving submission...")
    df_test["after"] = results

    # Ensure 'id' column exists as per submission format (sentence_id + "_" + token_id)
    # The provided data has sentence_id and token_id
    df_test["id"] = (
        df_test["sentence_id"].astype(str) + "_" + df_test["token_id"].astype(str)
    )

    submission_df = df_test[["id", "after"]]

    # Quote the 'after' column to match sample format exactly if needed,
    # but pandas to_csv handles quoting. The sample format shows quotes.
    # We use quoting=1 (QUOTE_ALL) or default (QUOTE_MINIMAL).
    # The sample provided: 0_0,"the" -> implies quoting strings.

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    return submission_df
