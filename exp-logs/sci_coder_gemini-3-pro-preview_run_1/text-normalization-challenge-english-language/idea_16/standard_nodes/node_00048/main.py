import sys
import os
import pandas as pd
import numpy as np
import torch
import warnings
import json
import torch.multiprocessing as mp
from scipy.stats import pearsonr

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration Override for Fast Baseline
# =============================================================================
from library.config import Config

# Reduce training epochs to ensure execution within time limits
Config.NUM_EPOCHS_TAGGER = 3
Config.NUM_EPOCHS_SEQ2SEQ = 3

# Aggressively subsample 'PLAIN' tokens to speed up Tagger training
# We keep 100% of 'interesting' sentences but only 2% of purely 'boring' ones.
Config.PLAIN_KEEP_RATE = 0.02

# =============================================================================
# Library Imports
# =============================================================================
from library.config import seed_everything
from library.dataset import build_artifacts, get_tagger_loader, Vocab
from library.trainer import run_tagger_safe, run_seq2seq_safe
from library.predictor import generate_submission
from library.models import MorphoBiLSTMTagger, CharSeq2Seq


# =============================================================================
# Validation Logic
# =============================================================================
def run_validation_pipeline():
    """
    Runs the full inference pipeline on the validation set to compute the
    exact competition metric (Accuracy) and perform failure analysis.
    """
    print("\n" + "=" * 40)
    print("Running Validation Pipeline")
    print("=" * 40)

    device = torch.device(Config.DEVICE)

    # 1. Load Validation Data
    # -----------------------
    print(f"Loading validation data from {Config.VAL_DATA_PATH}...")
    df_val = pd.read_csv(Config.VAL_DATA_PATH, keep_default_na=False)
    df_val["before"] = df_val["before"].astype(str)
    df_val["after"] = df_val["after"].astype(str)
    targets = df_val["after"].values

    # 2. Load Vocabularies
    # --------------------
    def load_json_vocab(filename):
        with open(os.path.join(Config.VOCAB_DIR, filename), "r") as f:
            return json.load(f)

    class_vocab = load_json_vocab("vocab_classes.json")
    seq2seq_vocab = load_json_vocab("vocab_seq2seq.json")
    word_vocab = load_json_vocab("vocab_words.json")
    char_vocab = load_json_vocab("vocab_chars.json")

    # Reverse maps for decoding
    id2class = {v: k for k, v in class_vocab.items()}
    id2char_seq2seq = {v: k for k, v in seq2seq_vocab.items()}

    # 3. Load Knowledge Base
    # ----------------------
    kb_path = os.path.join(Config.CACHE_DIR, "knowledge_base.parquet")
    kb = {}
    if os.path.exists(kb_path):
        print("Loading Knowledge Base...")
        kb_df = pd.read_parquet(kb_path)
        # Build dictionary for O(1) lookup
        for b, c, a in zip(kb_df["before"], kb_df["class"], kb_df["after"]):
            kb[(str(b), str(c))] = str(a)

    # 4. Load Models
    # --------------
    print("Loading trained models...")

    # Tagger
    tagger = MorphoBiLSTMTagger(
        word_vocab_size=len(word_vocab),
        class_vocab_size=len(class_vocab),
        char_vocab_size=len(char_vocab),
    ).to(device)
    tagger_path = os.path.join(Config.CHECKPOINT_DIR, "tagger_best_model.pth")
    tagger.load_state_dict(torch.load(tagger_path, map_location=device))
    tagger.eval()

    # Seq2Seq
    seq2seq = CharSeq2Seq(
        char_vocab_size=len(seq2seq_vocab), num_classes=len(class_vocab)
    ).to(device)
    seq2seq_path = os.path.join(Config.CHECKPOINT_DIR, "seq2seq_best_model.pth")
    seq2seq.load_state_dict(torch.load(seq2seq_path, map_location=device))
    seq2seq.eval()

    # 5. Inference Loop
    # -----------------
    # Use the Tagger Loader to handle batching and feature extraction
    val_loader = get_tagger_loader("val", batch_size=Config.BATCH_SIZE, shuffle=False)

    final_preds = [""] * len(df_val)

    # Buffers for OOV items requiring Seq2Seq
    oov_indices = []
    oov_tokens = []
    oov_classes = []

    print("Phase 1: Tagging and Retrieval...")
    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            word_ids = batch["word_ids"].to(device)
            char_features = batch["char_features"].to(device)
            regex_features = batch["regex_features"].to(device)
            lengths = batch["lengths"]

            # Predict Classes
            logits = tagger(word_ids, char_features, regex_features, lengths)
            preds = torch.argmax(logits, dim=2).cpu().numpy()

            # Map predictions back to original dataframe indices
            batch_orig_indices = batch["original_indices"]

            for i, sent_indices in enumerate(batch_orig_indices):
                # Get predictions for this sentence (ignoring padding)
                sent_preds = preds[i][: len(sent_indices)]

                for idx_tensor, class_idx in zip(sent_indices, sent_preds):
                    idx = idx_tensor.item()
                    token = df_val.at[idx, "before"]
                    class_name = id2class.get(class_idx, "PLAIN")

                    # Strategy: KB -> Copy (Plain) -> Seq2Seq
                    kb_key = (token, class_name)

                    if kb_key in kb:
                        final_preds[idx] = kb[kb_key]
                    elif class_name in ["PLAIN", "PUNCT"]:
                        final_preds[idx] = token
                    else:
                        # Queue for fallback
                        oov_indices.append(idx)
                        oov_tokens.append(token)
                        oov_classes.append(class_idx)

    # Phase 2: Seq2Seq Fallback
    print(f"Phase 2: Running Seq2Seq on {len(oov_indices)} OOV tokens...")
    if len(oov_indices) > 0:
        batch_size = Config.BATCH_SIZE
        num_samples = len(oov_indices)

        for i in range(0, num_samples, batch_size):
            batch_tokens = oov_tokens[i : i + batch_size]
            batch_class_idxs = oov_classes[i : i + batch_size]
            batch_indices = oov_indices[i : i + batch_size]

            # Prepare Seq2Seq Inputs
            src_ids_list = []
            unk_id = seq2seq_vocab.get("<UNK>")
            for t in batch_tokens:
                ids = [seq2seq_vocab.get(c, unk_id) for c in str(t)]
                src_ids_list.append(torch.tensor(ids, dtype=torch.long))

            src_lens = torch.tensor([len(s) for s in src_ids_list], dtype=torch.long)
            src_ids = torch.nn.utils.rnn.pad_sequence(
                src_ids_list, batch_first=True, padding_value=0
            ).to(device)

            class_ids = torch.tensor(batch_class_idxs, dtype=torch.long).to(device)

            # Generate
            with torch.no_grad():
                generated_ids = seq2seq(src_ids, src_lens, class_ids, tgt_ids=None)

            generated_ids = generated_ids.cpu().numpy()

            # Decode
            for j, gen_seq in enumerate(generated_ids):
                chars = []
                for char_id in gen_seq:
                    if char_id == 3:  # EOS
                        break
                    if char_id > 3:  # Skip specials
                        chars.append(id2char_seq2seq.get(char_id, ""))

                pred_str = "".join(chars)
                original_idx = batch_indices[j]
                final_preds[original_idx] = pred_str

    # 6. Compute Metric
    # -----------------
    correct_count = 0
    total_count = len(targets)
    error_flags = []

    for i in range(total_count):
        if final_preds[i] == targets[i]:
            correct_count += 1
            error_flags.append(0)
        else:
            error_flags.append(1)

    accuracy = correct_count / total_count
    print(f"Final Validation Metric: {accuracy}")

    # 7. Failure Analysis
    # -------------------
    print("\n--- Failure Analysis ---")
    df_val["is_error"] = error_flags
    df_val["token_len"] = df_val["before"].str.len()

    # Correlation between Error and Token Length
    if len(df_val) > 1 and df_val["is_error"].std() > 0:
        corr, _ = pearsonr(df_val["is_error"], df_val["token_len"])
        print(f"Correlation between Error and Token Length: {corr:.6f}")
    else:
        print("Correlation cannot be computed (no variance in errors).")

    # Error Rate by Class
    print("\nError Rate by Class (Top 5 Worst):")
    class_perf = df_val.groupby("class")["is_error"].agg(["mean", "count"])
    class_perf = class_perf[class_perf["count"] > 100]  # Filter rare classes
    print(class_perf.sort_values("mean", ascending=False).head(5))

    return accuracy


# =============================================================================
# Main Execution
# =============================================================================
def main():
    # Set Seed
    seed_everything(Config.SEED)

    # Define Overrides for Fast Baseline
    # These must be passed to child processes since they reload Config from disk
    overrides = {
        "NUM_EPOCHS_TAGGER": Config.NUM_EPOCHS_TAGGER,
        "NUM_EPOCHS_SEQ2SEQ": Config.NUM_EPOCHS_SEQ2SEQ,
        "PLAIN_KEEP_RATE": Config.PLAIN_KEEP_RATE,
        "SEED": Config.SEED,
    }

    # 1. Build Data Artifacts
    # This creates vocabs and the knowledge base from training data
    print("Step 1: Building Data Artifacts...")
    build_artifacts(load_cached_data=True)

    # 2. Train Tagger
    # The TaggerTrainer handles data loading (with density sampling) and training loop
    print("\nStep 2: Training Morphologically-Augmented Tagger (Isolated)...")
    ctx = mp.get_context("spawn")
    p1 = ctx.Process(target=run_tagger_safe, args=(overrides,))
    p1.start()
    p1.join()
    if p1.exitcode != 0:
        raise RuntimeError("Tagger Training Failed")

    # 3. Train Seq2Seq Fallback
    # The Seq2SeqTrainer trains only on 'changed' tokens
    print("\nStep 3: Training Seq2Seq Fallback Model (Isolated)...")
    p2 = ctx.Process(target=run_seq2seq_safe, args=(overrides,))
    p2.start()
    p2.join()
    if p2.exitcode != 0:
        if os.path.exists("seq2seq_error.log"):
            print("\n--- Seq2Seq Child Process Error Log ---")
            with open("seq2seq_error.log", "r") as f:
                print(f.read())
            print("---------------------------------------")
        raise RuntimeError("Seq2Seq Training Failed")

    # 4. Validation & Failure Analysis
    val_metric = run_validation_pipeline()

    # 5. Submission Generation
    # Only generate if we meet the strict threshold
    threshold = 0.9949142925818993

    if val_metric > threshold:
        print(f"\nValidation Metric ({val_metric}) exceeds threshold ({threshold}).")
        print("Generating Submission File...")
        generate_submission()
    else:
        print(
            f"\nValidation Metric ({val_metric}) did not exceed threshold ({threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
