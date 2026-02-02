import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from library
from library.config import Config
from library.utils import set_seed, get_device
from library.train_tagger import train_tagger_model
from library.train_seq2seq import train_seq2seq_model
from library.inference import generate_submission, InferenceSeq2SeqDataset
from library.data_processing import prepare_data, TaggerDataset
from library.models_tagger import QuadHybridBiLSTM
from library.models_seq2seq import CharTransformer

# ==============================================================================
# Configuration Override for Fast Baseline
# ==============================================================================
# We override default config values to ensure the script completes within 2 hours
# while still using the full dataset to maximize Knowledge Base coverage.
Config.NUM_EPOCHS = 1
Config.BATCH_SIZE = 256
Config.NUM_WORKERS = 8
Config.SAMPLE_SIZE = None  # Use full dataset
Config.DEBUG = False


def run_validation_and_analysis():
    """
    Runs the full inference pipeline on the validation set to compute the
    official metric and perform failure analysis.
    """
    config = Config()
    device = get_device()
    set_seed(config.SEED)

    print("\n" + "=" * 40)
    print("Running Validation & Failure Analysis")
    print("=" * 40)

    # 1. Load Resources
    # We use load_cached_data=True because training has already processed the data
    print("Loading validation data and artifacts...")
    artifacts = prepare_data(load_cached_data=True)

    vocab_words = artifacts["vocab_words"]
    vocab_chars = artifacts["vocab_chars"]
    vocab_classes = artifacts["vocab_classes"]
    bpe_tokenizer = artifacts["bpe_tokenizer"]
    val_grouped = artifacts["val_grouped"]
    kb_df = artifacts["kb_df"]

    # Build Knowledge Base Map for O(1) lookup
    print("Indexing Knowledge Base...")
    kb_map = dict(zip(zip(kb_df["before"], kb_df["class"]), kb_df["after"]))

    # 2. Load Models
    print("Loading trained models...")
    tagger = QuadHybridBiLSTM(
        num_classes=len(vocab_classes),
        vocab_words=vocab_words,
        vocab_chars=vocab_chars,
        vocab_bpe_size=config.BPE_VOCAB_SIZE,
    )
    if os.path.exists(config.TAGGER_MODEL_PATH):
        tagger.load_state_dict(
            torch.load(config.TAGGER_MODEL_PATH, map_location=device)
        )
    else:
        print("Error: Tagger model not found!")
        return 0.0
    tagger.to(device)
    tagger.eval()

    seq2seq = CharTransformer(
        vocab_chars_size=len(vocab_chars),
        vocab_classes_size=len(vocab_classes),
    )
    if os.path.exists(config.SEQ2SEQ_MODEL_PATH):
        seq2seq.load_state_dict(
            torch.load(config.SEQ2SEQ_MODEL_PATH, map_location=device)
        )
    else:
        print("Error: Seq2Seq model not found!")
        return 0.0
    seq2seq.to(device)
    seq2seq.eval()

    # 3. Tagger Inference
    print("Running Tagger on Validation Set...")
    val_dataset = TaggerDataset(
        val_grouped, vocab_words, vocab_chars, vocab_classes, bpe_tokenizer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    flat_data = []
    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Tagging"):
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            bpe_ids = batch["bpe_ids"].to(device)
            features = batch["features"].to(device)

            # Forward pass
            logits = tagger(word_ids, char_ids, bpe_ids, features)
            preds = torch.argmax(logits, dim=2).cpu().numpy()  # (Batch, Seq)

            batch_size = word_ids.size(0)

            # Map predictions back to tokens
            for b in range(batch_size):
                row = val_grouped.iloc[global_idx]
                tokens = row["before"]
                true_classes = row["class"]
                true_afters = row["after"]

                # Handle truncation/padding
                sent_len = min(len(tokens), config.MAX_SENT_LEN)
                sent_preds = preds[b][:sent_len]

                for k in range(sent_len):
                    token = tokens[k]
                    pred_cls_idx = sent_preds[k]
                    pred_cls = vocab_classes.get_token(pred_cls_idx)
                    true_after = true_afters[k]

                    flat_data.append(
                        {
                            "token": token,
                            "pred_class": pred_cls,
                            "true_after": true_after,
                            "true_class": true_classes[k],
                        }
                    )

                global_idx += 1

    # 4. Hybrid Logic & Seq2Seq Fallback
    print("Applying Hybrid Logic and Seq2Seq Fallback...")

    seq2seq_indices = []
    seq2seq_tokens = []
    seq2seq_classes = []

    final_predictions = [""] * len(flat_data)

    for i, item in enumerate(flat_data):
        token = item["token"]
        cls = item["pred_class"]

        # Strategy 1: Knowledge Base Lookup
        if (token, cls) in kb_map:
            final_predictions[i] = kb_map[(token, cls)]
        # Strategy 2: Copy if PLAIN/PUNCT
        elif cls == "PLAIN" or cls == "PUNCT":
            final_predictions[i] = token
        # Strategy 3: Neural Generation (Fallback)
        else:
            seq2seq_indices.append(i)
            seq2seq_tokens.append(token)
            seq2seq_classes.append(cls)

    # Run Seq2Seq for OOV tokens
    if seq2seq_indices:
        print(f"Generating text for {len(seq2seq_indices)} OOV tokens...")
        s2s_dataset = InferenceSeq2SeqDataset(
            seq2seq_tokens, seq2seq_classes, vocab_chars, vocab_classes
        )
        s2s_loader = DataLoader(
            s2s_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

        generated_texts = []
        sos_id = vocab_chars.get_id("<SOS>")
        eos_id = vocab_chars.get_id("<EOS>")

        with torch.no_grad():
            for batch in tqdm(s2s_loader, desc="Seq2Seq"):
                src_ids = batch["src_ids"].to(device)
                class_id = batch["class_id"].to(device)

                output_ids = seq2seq.predict(
                    src_ids,
                    class_id,
                    max_len=config.MAX_TOKEN_CHAR_LEN,
                    sos_id=sos_id,
                    eos_id=eos_id,
                )

                output_ids = output_ids.cpu().numpy()
                for seq in output_ids:
                    chars = []
                    for idx in seq:
                        if idx == sos_id:
                            continue
                        if idx == eos_id:
                            break
                        if idx == vocab_chars.get_id("<PAD>"):
                            continue
                        chars.append(vocab_chars.get_token(idx))
                    generated_texts.append("".join(chars))

        # Fill back predictions
        for idx, text in zip(seq2seq_indices, generated_texts):
            final_predictions[idx] = text

    # 5. Compute Metric
    targets = [x["true_after"] for x in flat_data]

    correct_count = sum(p == t for p, t in zip(final_predictions, targets))
    total_count = len(targets)
    accuracy = correct_count / total_count

    print(f"Final Validation Metric: {accuracy}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = pd.DataFrame(flat_data)
    df_analysis["pred_after"] = final_predictions
    df_analysis["is_error"] = (
        df_analysis["true_after"] != df_analysis["pred_after"]
    ).astype(int)

    # Compute features for correlation
    df_analysis["token_len"] = df_analysis["token"].apply(len)
    df_analysis["is_digit"] = df_analysis["token"].str.contains(r"\d").astype(int)

    # Calculate correlations
    correlations = df_analysis[["is_error", "token_len", "is_digit"]].corr()["is_error"]
    print("Correlation between Error and Features:")
    print(correlations)

    return accuracy


def main():
    # 1. Train Tagger
    print("\n" + "=" * 40)
    print("Step 1: Training Tagger")
    print("=" * 40)
    train_tagger_model(load_cached_data=True)

    # 2. Train Seq2Seq
    print("\n" + "=" * 40)
    print("Step 2: Training Seq2Seq Fallback")
    print("=" * 40)
    train_seq2seq_model(load_cached_data=True)

    # 3. Validation & Analysis
    val_acc = run_validation_and_analysis()

    # 4. Submission
    threshold = 0.9949142925818993
    if val_acc > threshold:
        print(f"\nValidation accuracy ({val_acc}) exceeds threshold ({threshold}).")
        print("Generating submission file...")
        generate_submission()
    else:
        print(
            f"\nValidation accuracy ({val_acc}) does not meet threshold ({threshold})."
        )
        print("Submission skipped.")


if __name__ == "__main__":
    main()
