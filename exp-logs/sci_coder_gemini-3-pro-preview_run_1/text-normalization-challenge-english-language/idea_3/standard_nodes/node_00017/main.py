import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Import library modules
from library.config import Config
from library.dataset import (
    get_tagger_dataloaders,
    get_seq2seq_dataloaders,
    get_test_dataloader,
    get_knowledge_base,
    load_or_create_vocabs,
    load_or_create_grouped_data,
)
from library.models import BiLSTMTagger, Seq2SeqNormalizer
from library.trainer import train_tagger, train_seq2seq
from library.utils import set_seed, load_checkpoint

# -------------------------------------------------------------------------
# 1. CONFIGURATION OVERRIDE
# -------------------------------------------------------------------------
# Override Config for a fast baseline run within time limits
Config.EPOCHS = 2  # Reduce epochs to ensure completion within 2 hours
Config.DEBUG = False  # Use full dataset for best accuracy (A100 is fast enough)
Config.BATCH_SIZE = 256  # Increase batch size for A100 efficiency
Config.NUM_WORKERS = 4

# -------------------------------------------------------------------------
# 2. INFERENCE UTILITIES
# -------------------------------------------------------------------------


def run_tagger_inference(model, loader, device):
    """
    Runs the tagger on the loader and returns a list of arrays of predicted class indices.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Tagger Inference", leave=False):
            # Unpack batch (handle test vs train/val structure)
            if len(batch) == 3:
                if isinstance(
                    batch[2], list
                ):  # Test loader: word_ids, char_ids, row_ids
                    word_ids, char_ids, _ = batch
                else:  # Train/Val loader: word_ids, char_ids, class_ids
                    word_ids, char_ids, _ = batch
            else:
                raise ValueError("Unexpected batch format")

            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)

            # Forward
            logits = model(word_ids, char_ids)  # (batch, seq, classes)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


def run_seq2seq_batch(model, tokens, vocab_chars, device):
    """
    Runs Seq2Seq inference on a list of raw tokens.
    """
    model.eval()
    results = []

    # Process in chunks to avoid OOM
    chunk_size = 512

    sos_idx = vocab_chars.stoi["<sos>"]
    eos_idx = vocab_chars.stoi["<eos>"]

    with torch.no_grad():
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i : i + chunk_size]

            # Convert to indices
            src_list = []
            for t in chunk:
                chars = list(str(t))
                ids = vocab_chars.lookup_indices(chars)
                src_list.append(torch.tensor(ids, dtype=torch.long))

            # Pad
            src_padded = pad_sequence(src_list, batch_first=True, padding_value=0).to(
                device
            )

            # Predict
            preds = model.predict(src_padded, sos_idx, eos_idx)

            # Decode
            for j in range(len(chunk)):
                pred_ids = preds[j].cpu().numpy()
                decoded_chars = []
                for idx in pred_ids:
                    if idx == eos_idx:
                        break
                    if idx == 0:  # pad
                        continue
                    decoded_chars.append(vocab_chars.lookup_token(idx))

                results.append("".join(decoded_chars))

    return results


def full_inference(df, loader, tagger, seq2seq, kb, vocab_classes, vocab_chars, device):
    """
    Orchestrates the cascade inference: Tagger -> KB -> Seq2Seq.
    Returns list of [id, predicted_text].
    """
    # 1. Run Tagger
    tagger_preds_padded = run_tagger_inference(tagger, loader, device)

    # 2. Flatten and Align
    final_predictions = []  # List of [id, text]

    # Collect tokens needing Seq2Seq for batched processing
    # Store as (index_in_final_predictions, token_text)
    seq2seq_requests = []

    # Pre-calculate class map
    idx_to_class = vocab_classes.itos

    # Ensure alignment
    if len(df) != len(tagger_preds_padded):
        print(
            f"Warning: DF length {len(df)} != Tagger Output {len(tagger_preds_padded)}"
        )

    global_idx = 0

    # Iterate sentences
    for i in tqdm(range(len(df)), desc="Processing Sentences"):
        row = df.iloc[i]
        tokens = row["before"]
        ids = row["id"]

        # Get preds for this sentence, slicing to actual length
        seq_len = len(tokens)
        sent_preds = tagger_preds_padded[i][:seq_len]

        for j, (token, pred_idx, token_id) in enumerate(zip(tokens, sent_preds, ids)):
            pred_class = idx_to_class.get(pred_idx, "PLAIN")

            norm_text = None

            # Tier 1: Identity
            if pred_class in ["PLAIN", "PUNCT"]:
                norm_text = token
            else:
                # Tier 2: Knowledge Base
                kb_val = kb.get(token, pred_class)
                if kb_val is not None:
                    norm_text = kb_val
                else:
                    # Tier 3: Seq2Seq (Deferred)
                    seq2seq_requests.append((global_idx, token))
                    norm_text = ""  # Placeholder

            final_predictions.append([token_id, norm_text])
            global_idx += 1

    # 3. Run Seq2Seq for misses
    if seq2seq_requests:
        print(f"Running Seq2Seq fallback for {len(seq2seq_requests)} tokens...")
        indices, req_tokens = zip(*seq2seq_requests)

        # Run batch inference
        generated_texts = run_seq2seq_batch(seq2seq, req_tokens, vocab_chars, device)

        # Fill back results
        for idx, text in zip(indices, generated_texts):
            final_predictions[idx][1] = text

    return final_predictions


# -------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -------------------------------------------------------------------------


def main():
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on {device}")

    # --- A. Training ---
    # Train Tagger
    train_tagger(load_cached_data=True)

    # Train Seq2Seq
    train_seq2seq(load_cached_data=True)

    # --- B. Load Resources for Inference ---
    # Load Vocabs (Should be cached by training steps)
    vocab_tokens, vocab_chars, vocab_classes = load_or_create_vocabs(
        None, load_cached_data=True
    )

    # Load KB
    kb = get_knowledge_base(load_cached_data=True)

    # Load Models
    tagger = BiLSTMTagger(len(vocab_tokens), len(vocab_chars), len(vocab_classes)).to(
        device
    )
    load_checkpoint(Config.TAGGER_MODEL_PATH, tagger, device=device)

    seq2seq = Seq2SeqNormalizer(len(vocab_chars)).to(device)
    load_checkpoint(Config.SEQ2SEQ_MODEL_PATH, seq2seq, device=device)

    # --- C. Validation ---
    print("\nStarting Validation...")
    # Load Val Data
    val_df = load_or_create_grouped_data(
        Config.VAL_DATA_PATH, Config.VAL_GROUPED_PATH, load_cached_data=True
    )

    # Get Val Loader (reuse function to get consistent loader)
    # Note: get_tagger_dataloaders returns (train, val, ...). We discard train.
    _, val_loader, _, _, _ = get_tagger_dataloaders(load_cached_data=True)

    # Run Inference
    val_preds = full_inference(
        val_df, val_loader, tagger, seq2seq, kb, vocab_classes, vocab_chars, device
    )

    # Calculate Metric
    # Flatten targets from val_df
    val_targets = [t for seq in val_df["after"] for t in seq]
    val_ids = [i for seq in val_df["id"] for i in seq]

    correct_count = 0
    total_count = len(val_targets)

    errors = []  # For failure analysis: (0 for correct, 1 for error)

    print("Calculating Accuracy...")
    for i in range(total_count):
        pred_id, pred_text = val_preds[i]
        true_text = val_targets[i]
        true_id = val_ids[i]

        # Sanity check alignment
        if pred_id != true_id:
            print(f"Mismatch IDs at {i}: {pred_id} vs {true_id}")
            continue

        is_correct = pred_text == true_text
        if is_correct:
            correct_count += 1

        errors.append(int(not is_correct))

    accuracy = correct_count / total_count
    print(f"Final Validation Metric: {accuracy}")

    # --- D. Failure Analysis ---
    print("\nFailure Analysis:")
    # Correlation between error and token length
    # Re-iterate to get lengths
    token_lengths = [len(str(t)) for seq in val_df["before"] for t in seq]

    if len(token_lengths) == len(errors):
        corr = np.corrcoef(token_lengths, errors)[0, 1]
        print(f"Correlation between Error and Token Length: {corr:.4f}")
    else:
        print("Could not calculate correlation due to length mismatch.")

    # --- E. Submission ---
    THRESHOLD = 0.9861543320467205
    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy {accuracy:.5f} > {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        test_loader, test_df, _, _, _ = get_test_dataloader(load_cached_data=True)

        # Run Inference
        test_preds = full_inference(
            test_df,
            test_loader,
            tagger,
            seq2seq,
            kb,
            vocab_classes,
            vocab_chars,
            device,
        )

        # Create DataFrame
        sub_df = pd.DataFrame(test_preds, columns=["id", "after"])

        # Save
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation accuracy {accuracy:.5f} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
