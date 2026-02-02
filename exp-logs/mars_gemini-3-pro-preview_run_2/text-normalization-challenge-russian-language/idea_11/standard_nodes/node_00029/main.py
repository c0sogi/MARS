import os
import sys
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, load_raw_data, get_device, cleanup
from library.tokenization import build_tokenizers
from library.hfbb_layer import HFBBModel
from library.trainer import train_model, greedy_decode, generate_submission
from library.data_manager import _add_context_columns
from library.transformer_arch import CharToSubwordTransformer


def run_hybrid_inference(df, hfbb, model, char_tokenizer, bpe_tokenizer, device):
    """
    Runs the hybrid inference pipeline (HFBB + Transformer) on a dataframe.
    Returns a list of predictions.
    """
    # Initialize results with None
    results = [None] * len(df)
    transformer_indices = []

    # Pre-calculate semiotic mask
    semiotic_mask = (
        df["before"].astype(str).str.contains(Config.SEMIOTIC_REGEX, regex=True)
    )

    # Convert columns to lists for fast iteration
    befores = df["before"].astype(str).tolist()
    prev1s = df["prev_1"].fillna("").astype(str).tolist()
    next1s = df["next_1"].fillna("").astype(str).tolist()

    # Tier 1: HFBB & Routing
    print("Executing Tier 1 (HFBB) and Routing...")
    for i in range(len(df)):
        pred = hfbb.query(befores[i], prev1s[i], next1s[i])
        if pred is not None:
            results[i] = pred
        else:
            if semiotic_mask.iloc[i]:
                transformer_indices.append(i)
            else:
                results[i] = befores[i]

    # Tier 2: Transformer
    if transformer_indices:
        print(f"Executing Tier 2 (Transformer) on {len(transformer_indices)} tokens...")
        batch_size = Config.BATCH_SIZE
        num_batches = (len(transformer_indices) + batch_size - 1) // batch_size

        sep_id = char_tokenizer.sep_token_id
        pad_id = char_tokenizer.pad_token_id

        # Helper for encoding
        def encode_text(text):
            return char_tokenizer.encode(str(text), add_special_tokens=False)

        space_ids = encode_text(" ")

        for b in range(num_batches):
            batch_idxs = transformer_indices[b * batch_size : (b + 1) * batch_size]
            src_batch = []

            # Prepare batch tensors
            for idx in batch_idxs:
                row = df.iloc[idx]
                p2 = str(row.get("prev_2", ""))
                p1 = str(row.get("prev_1", ""))
                curr = str(row["before"])
                n1 = str(row.get("next_1", ""))
                n2 = str(row.get("next_2", ""))

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
                src_batch, batch_first=True, padding_value=pad_id
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

            # Decode
            for k, g_ids in enumerate(generated_ids):
                ids_list = g_ids.cpu().tolist()
                decoded_text = bpe_tokenizer.decode(ids_list, skip_special_tokens=True)
                results[batch_idxs[k]] = decoded_text

            if (b + 1) % 500 == 0:
                cleanup()

    return results


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Tokenization
    print("Building/Loading Tokenizers...")
    char_tokenizer, bpe_tokenizer = build_tokenizers(load_cached_data=True)

    # 3. HFBB Initialization (Tier 1)
    print("Initializing HFBB Model...")
    hfbb = HFBBModel(load_cached_data=True)

    # 4. Transformer Training (Tier 2)
    # Fast baseline: Train on 100k samples for 2 epochs
    print("Starting Transformer Training (Fast Baseline)...")
    train_model(char_tokenizer, bpe_tokenizer, num_epochs=2, debug_subset_size=100000)

    # 5. Validation
    print("Loading Validation Data...")
    df_val = load_raw_data("val")
    df_val = _add_context_columns(df_val)

    # Load Best Model
    print("Loading Best Transformer Checkpoint...")
    model = CharToSubwordTransformer(
        src_vocab_size=len(char_tokenizer),
        tgt_vocab_size=len(bpe_tokenizer),
        src_pad_idx=char_tokenizer.pad_token_id,
        tgt_pad_idx=bpe_tokenizer.pad_token_id,
    ).to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: No checkpoint found. Validation will use random weights.")
    model.eval()

    # Run Inference
    print("Running Hybrid Validation Inference...")
    predictions = run_hybrid_inference(
        df_val, hfbb, model, char_tokenizer, bpe_tokenizer, device
    )

    df_val["pred"] = predictions
    # Fill any potential Nones with identity (fallback)
    df_val["pred"] = df_val["pred"].fillna(df_val["before"])

    # Calculate Metric
    correct_mask = df_val["pred"] == df_val["after"]
    accuracy = correct_mask.mean()

    print(f"Final Validation Metric: {accuracy}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    df_val["is_error"] = ~correct_mask

    # Error Rate by Class
    print("Error Rate by Class:")
    class_errors = (
        df_val.groupby("class")["is_error"].mean().sort_values(ascending=False)
    )
    print(class_errors.head(10))

    # Correlation with Input Length
    df_val["len_before"] = df_val["before"].astype(str).apply(len)
    correlation = df_val["is_error"].corr(df_val["len_before"])
    print(f"\nCorrelation (Error vs Input Length): {correlation:.4f}")

    # 7. Submission
    SUBMISSION_THRESHOLD = 0.9788071831831453
    if accuracy > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation accuracy {accuracy} > {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        # Clean up memory before full inference
        del df_val, model, hfbb
        cleanup()
        generate_submission(char_tokenizer, bpe_tokenizer)
    else:
        print(
            f"\nValidation accuracy {accuracy} <= {SUBMISSION_THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
