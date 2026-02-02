import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import nltk

# Import provided library components
from library.config import Config
from library.vocab import Vocabulary
from library.data import InterleavedDataset, collate_fn
from library.model import BifurcatedTransformer
from library.inference import Predictor

# ------------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# ------------------------------------------------------------------------------
Config.EPOCHS = 1
Config.BATCH_SIZE = 128
TRAIN_SAMPLES = 200000  # Limit training data for speed
VAL_SAMPLES = 5000  # Limit validation data for speed
THRESHOLD_METRIC = 7.214528751275944


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


def run_training(device, vocab):
    print(f"--- Starting Training (Max Samples: {TRAIN_SAMPLES}) ---")

    # Initialize Model
    model = BifurcatedTransformer().to(device)
    model.train()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Loss Functions
    # Localization: BCE with positive weight for class imbalance
    loc_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([20.0]).to(device), reduction="none"
    )
    # Identification: Cross Entropy
    id_criterion = nn.CrossEntropyLoss()

    # Load Data
    train_dataset = InterleavedDataset(
        "train", vocab, load_cached_data=True, max_samples=TRAIN_SAMPLES
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Training Loop
    total_steps = len(train_loader)
    print(
        f"Training for {Config.EPOCHS} epoch(s) with {total_steps} steps per epoch..."
    )

    for epoch in range(Config.EPOCHS):
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_loc = batch["target_loc"].to(device)
            target_word = batch["target_word"].to(device)

            optimizer.zero_grad()

            # Forward
            loc_logits, id_logits = model(input_ids, attention_mask)

            # --- Localization Loss ---
            valid_mask = target_loc >= 0
            batch_size, seq_len = input_ids.shape

            # Create binary target mask
            loc_targets = torch.zeros((batch_size, seq_len), device=device)
            safe_target_loc = target_loc.clone()
            safe_target_loc[~valid_mask] = 0
            loc_targets.scatter_(1, safe_target_loc.unsqueeze(1), 1.0)

            loc_logits_flat = loc_logits.squeeze(-1)
            bce_loss = loc_criterion(loc_logits_flat, loc_targets)

            # Mask padding and invalid samples
            loss_mask = attention_mask.float() * valid_mask.unsqueeze(1).float()
            loc_loss = (bce_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)

            # --- Identification Loss ---
            valid_indices = torch.nonzero(valid_mask).squeeze(-1)
            if valid_indices.numel() > 0:
                v_target_loc = target_loc[valid_indices]
                v_target_word = target_word[valid_indices]
                v_id_logits = id_logits[valid_indices]

                # Gather logits at gap index
                gather_index = v_target_loc.view(-1, 1, 1).expand(
                    -1, 1, Config.VOCAB_SIZE
                )
                selected_logits = torch.gather(v_id_logits, 1, gather_index).squeeze(1)

                id_loss = id_criterion(selected_logits, v_target_word)
            else:
                id_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Total Loss
            loss = Config.LAMBDA_LOC * loc_loss + Config.LAMBDA_ID * id_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / steps:.4f}")

    # Save Model
    print(f"Saving model to {Config.MODEL_PATH}")
    torch.save(model.state_dict(), Config.MODEL_PATH)
    return model


def reconstruct_sentence(vocab, token_ids, gap_idx, word_idx):
    """Helper to reconstruct sentence from interleaved tokens + gap + word."""
    # Decode predicted word
    word = vocab.itos.get(word_idx, Config.UNK_TOKEN)

    # Extract original words from interleaved sequence (even indices)
    words = []
    for idx, tid in enumerate(token_ids):
        if tid == 0:  # PAD
            break
        if idx % 2 == 0:
            w = vocab.itos.get(tid, Config.UNK_TOKEN)
            words.append(w)

    # Insert word
    # Gap at index g (odd) is between word (g-1)/2 and (g+1)/2
    insert_idx = (gap_idx + 1) // 2
    if insert_idx > len(words):
        insert_idx = len(words)

    words.insert(insert_idx, word)
    return " ".join(words)


def validate_and_analyze(device, vocab):
    print(f"--- Starting Validation (Samples: {VAL_SAMPLES}) ---")

    # Load Model
    model = BifurcatedTransformer().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Load Validation Data
    val_dataset = InterleavedDataset(
        "val", vocab, load_cached_data=True, max_samples=VAL_SAMPLES
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    levenshtein_scores = []
    lengths = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_loc = batch["target_loc"].to(device)
            target_word = batch["target_word"].to(device)

            # Filter invalid samples (truncated targets)
            valid_mask = target_loc >= 0
            if not valid_mask.any():
                continue

            # Forward
            loc_logits, id_logits = model(input_ids, attention_mask)

            # Probabilities
            loc_probs = torch.sigmoid(loc_logits).squeeze(-1)
            id_probs = torch.softmax(id_logits, dim=-1)

            # Structural Masking (Odd indices only)
            seq_len = input_ids.shape[1]
            gap_mask = torch.zeros((seq_len,), device=device)
            gap_mask[1::2] = 1.0

            loc_probs = loc_probs * gap_mask.unsqueeze(0) * attention_mask

            # Score Fusion
            scores = loc_probs.unsqueeze(-1) * id_probs

            # Predictions
            batch_size = input_ids.shape[0]
            scores_flat = scores.view(batch_size, -1)
            best_flat_indices = torch.argmax(scores_flat, dim=1)

            pred_gap_indices = best_flat_indices // Config.VOCAB_SIZE
            pred_word_indices = best_flat_indices % Config.VOCAB_SIZE

            # CPU conversion for reconstruction
            input_ids_cpu = input_ids.cpu().numpy()
            target_loc_cpu = target_loc.cpu().numpy()
            target_word_cpu = target_word.cpu().numpy()
            pred_gap_cpu = pred_gap_indices.cpu().numpy()
            pred_word_cpu = pred_word_indices.cpu().numpy()
            valid_mask_cpu = valid_mask.cpu().numpy()

            for k in range(batch_size):
                if not valid_mask_cpu[k]:
                    continue

                # Reconstruct Ground Truth
                gt_sent = reconstruct_sentence(
                    vocab, input_ids_cpu[k], target_loc_cpu[k], target_word_cpu[k]
                )

                # Reconstruct Prediction
                pred_sent = reconstruct_sentence(
                    vocab, input_ids_cpu[k], pred_gap_cpu[k], pred_word_cpu[k]
                )

                # Calculate Levenshtein
                dist = nltk.edit_distance(gt_sent, pred_sent)
                levenshtein_scores.append(dist)
                lengths.append(len(gt_sent))

    # Metric Calculation
    final_metric = np.mean(levenshtein_scores)

    # Failure Analysis
    print("--- Failure Analysis ---")
    if len(lengths) > 1:
        correlation = np.corrcoef(lengths, levenshtein_scores)[0, 1]
        print(
            f"Correlation between Sentence Length and Error (Levenshtein): {correlation:.4f}"
        )
    else:
        print("Not enough samples for correlation analysis.")

    return final_metric


def main():
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Build Vocabulary
    print("Building Vocabulary...")
    vocab = Vocabulary()
    vocab.build(load_cached_data=True)

    # Train
    run_training(device, vocab)

    # Validate
    metric = validate_and_analyze(device, vocab)
    print(f"Final Validation Metric: {metric}")

    # Submission
    if metric < THRESHOLD_METRIC:
        print("Metric threshold satisfied. Generating submission...")
        predictor = Predictor()
        predictor.predict()
    else:
        print(f"Metric {metric} >= {THRESHOLD_METRIC}. Submission skipped.")


if __name__ == "__main__":
    main()
