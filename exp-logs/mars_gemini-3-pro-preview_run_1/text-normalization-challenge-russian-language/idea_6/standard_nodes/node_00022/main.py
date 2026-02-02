import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config, DEVICE, PAD_TOKEN
from library.utils import seed_everything, load_metadata, calculate_accuracy
from library.data_manager import DataManager
from library.tokenizers import HybridTokenizer
from library.symbolic_model import HierarchicalNgram
from library.neural_arch import DualGranularityTransformer
from library.neural_dataset import NormalizationDataset, NormalizationCollator
from library.neural_trainer import ModelTrainer
from library.inference_engine import HybridRouter


def main():
    # --- 1. Configuration & Setup ---
    config = Config()
    # Adjust for fast baseline execution
    config.epochs = 5
    config.batch_size = 512  # A100 allows larger batch size
    config.num_workers = 4

    seed_everything(config.seed)
    print(f"Running on device: {DEVICE}")

    # --- 2. Data Preparation ---
    print("\n=== Data Preparation ===")
    dm = DataManager(config)

    # Reconstruct sentences (needed for context extraction)
    # This caches the result, so subsequent calls are fast
    print("Reconstructing sentences...")
    _ = dm.reconstruct_sentences("train")
    _ = dm.reconstruct_sentences("val")

    # Load raw training data for tokenizer and symbolic stats
    raw_train_df = load_metadata("train", config)

    # Train Tokenizers
    print("Training Tokenizers...")
    tokenizer = HybridTokenizer(config)
    tokenizer.train_tokenizers(raw_train_df)

    # Prepare Neural Sequences (Filtered dataset)
    print("Preparing Neural Sequences...")
    train_seq_df = dm.prepare_neural_sequences("train")
    val_seq_df = dm.prepare_neural_sequences("val")

    # --- 3. Symbolic Model Building ---
    print("\n=== Building Symbolic Model ===")
    symbolic_model = HierarchicalNgram(config)
    symbolic_model.build_stats(train_df=raw_train_df)

    # --- 4. Neural Model Training ---
    print("\n=== Training Neural Model ===")

    # Create Datasets
    train_dataset = NormalizationDataset(train_seq_df, tokenizer, config)
    val_dataset = NormalizationDataset(val_seq_df, tokenizer, config)

    collator = NormalizationCollator(
        bpe_pad_id=tokenizer.pad_token_id,
        char_pad_id=tokenizer.char_tokenizer.pad_token_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = DualGranularityTransformer(
        config,
        bpe_pad_id=tokenizer.pad_token_id,
        char_pad_id=tokenizer.char_tokenizer.pad_token_id,
    )

    # Train
    trainer = ModelTrainer(config, model, tokenizer)
    trainer.train(train_loader, val_loader)

    # Load best model for validation
    print("Loading best model checkpoint for validation...")
    best_state = torch.load(config.model_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(best_state)
    model.eval()
    model.to(DEVICE)

    # --- 5. Full Validation (Hybrid System) ---
    print("\n=== Performing Full Hybrid Validation ===")

    # Load full validation metadata
    df_val_full = load_metadata("val", config)
    # Ensure sorted
    df_val_full = df_val_full.sort_values(["sentence_id", "token_id"]).reset_index(
        drop=True
    )

    # Prepare Contexts for Symbolic Lookup
    tokens = df_val_full["before"].astype(str).values
    sent_ids = df_val_full["sentence_id"].values

    # Vectorized shift for context
    prev_tokens = np.roll(tokens, 1)
    prev_sent = np.roll(sent_ids, 1)
    prev_tokens[sent_ids != prev_sent] = PAD_TOKEN
    prev_tokens[0] = PAD_TOKEN

    next_tokens = np.roll(tokens, -1)
    next_sent = np.roll(sent_ids, -1)
    next_tokens[sent_ids != next_sent] = PAD_TOKEN
    next_tokens[-1] = PAD_TOKEN

    # Initialize predictions array
    final_preds = np.array([None] * len(df_val_full), dtype=object)

    # A. Priority 1: Trigram Lookup
    print("Applying Trigram Lookup...")
    trigram_keys = zip(prev_tokens, tokens, next_tokens)
    trigram_preds = [symbolic_model.trigram_stats.get(k) for k in trigram_keys]

    for i, p in enumerate(trigram_preds):
        if p is not None:
            final_preds[i] = p

    # B. Priority 2: Neural Model (Digits + Unsolved)
    mask_unsolved = final_preds == None
    mask_digits = df_val_full["before"].astype(str).str.contains(r"\d", regex=True)
    mask_neural = mask_unsolved & mask_digits

    if mask_neural.sum() > 0:
        print(f"Routing {mask_neural.sum()} tokens to Neural Model...")

        # Construct context for neural inference (Window +/- 2)
        l2_tokens = np.roll(tokens, 2)
        l2_sent = np.roll(sent_ids, 2)
        l2_tokens[sent_ids != l2_sent] = PAD_TOKEN
        l2_tokens[0] = PAD_TOKEN
        l2_tokens[1] = PAD_TOKEN

        r2_tokens = np.roll(tokens, -2)
        r2_sent = np.roll(sent_ids, -2)
        r2_tokens[sent_ids != r2_sent] = PAD_TOKEN
        r2_tokens[-1] = PAD_TOKEN
        r2_tokens[-2] = PAD_TOKEN

        indices = np.where(mask_neural)[0]

        # Create temporary dataframe for neural validation
        neural_val_data = []
        for idx in indices:
            neural_val_data.append(
                {
                    "context_left": [l2_tokens[idx], prev_tokens[idx]],
                    "before": tokens[idx],
                    "context_right": [next_tokens[idx], r2_tokens[idx]],
                    "id": idx,  # Use index as ID to map back
                }
            )

        df_neural_val = pd.DataFrame(neural_val_data)

        # Dataset & Loader
        ds_neural_val = NormalizationDataset(df_neural_val, tokenizer, config)
        dl_neural_val = DataLoader(
            ds_neural_val,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.num_workers,
        )

        # Inference Loop
        neural_results = []
        result_indices = []

        sos_id = tokenizer.bpe_tokenizer.token_to_id("<SOS>")
        eos_id = tokenizer.bpe_tokenizer.token_to_id("<EOS>")

        with torch.no_grad():
            for batch in dl_neural_val:
                src_left = batch["src_left"].to(DEVICE)
                src_target = batch["src_target"].to(DEVICE)
                src_right = batch["src_right"].to(DEVICE)
                batch_ids = batch["id"]

                # Greedy Decode
                batch_size = src_left.size(0)
                memory, memory_mask = model.encode(src_left, src_target, src_right)

                ys = torch.full(
                    (batch_size, 1), sos_id, dtype=torch.long, device=DEVICE
                )
                finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

                for _ in range(config.max_seq_len):
                    out = model.decode(ys, memory, memory_key_padding_mask=memory_mask)
                    prob = out[:, -1, :]
                    _, next_word = torch.max(prob, dim=1)
                    ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                    finished |= next_word == eos_id
                    if finished.all():
                        break

                # Decode strings
                ys_list = ys.tolist()
                for row in ys_list:
                    toks = []
                    for tid in row[1:]:
                        if tid == eos_id:
                            break
                        toks.append(tid)
                    neural_results.append(
                        tokenizer.decode(toks, skip_special_tokens=True)
                    )

                result_indices.extend(batch_ids)

        # Map results back
        for idx, res in zip(result_indices, neural_results):
            final_preds[idx] = res

    # C. Priority 3: Backoff (Bigram -> Unigram -> Identity)
    mask_remaining = final_preds == None
    if mask_remaining.sum() > 0:
        print(f"Backing off for {mask_remaining.sum()} tokens...")
        rem_indices = np.where(mask_remaining)[0]
        for idx in rem_indices:
            # Bigram
            bi = symbolic_model.bigram_stats.get((prev_tokens[idx], tokens[idx]))
            if bi is not None:
                final_preds[idx] = bi
                continue
            # Unigram
            uni = symbolic_model.unigram_stats.get(tokens[idx])
            if uni is not None:
                final_preds[idx] = uni
                continue
            # Identity
            final_preds[idx] = tokens[idx]

    # --- Calculate Metric ---
    targets = df_val_full["after"].astype(str).values
    final_preds_str = [str(p) if p is not None else "" for p in final_preds]

    accuracy = calculate_accuracy(final_preds_str, targets)
    print(f"Final Validation Metric: {accuracy:.20f}")

    # --- 6. Failure Analysis ---
    print("\n=== Failure Analysis ===")
    df_val_full["pred"] = final_preds_str
    df_val_full["correct"] = (df_val_full["pred"] == df_val_full["after"]).astype(int)

    # Correlation with Length
    df_val_full["len"] = df_val_full["before"].astype(str).str.len()
    len_corr = df_val_full["len"].corr(df_val_full["correct"])
    print(f"Correlation between Token Length and Correctness: {len_corr:.4f}")

    # Accuracy by Class
    print("\nAccuracy by Class:")
    class_acc = df_val_full.groupby("class")["correct"].mean().sort_values()
    print(class_acc)

    # --- 7. Submission ---
    threshold = 0.9859590499865803
    if accuracy > threshold:
        print(
            f"\nValidation accuracy {accuracy} > {threshold}. Generating submission..."
        )
        # Use the dedicated Inference Engine which handles test set loading and formatting
        router = HybridRouter(config)
        router.generate_submission()
    else:
        print(f"\nValidation accuracy {accuracy} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
