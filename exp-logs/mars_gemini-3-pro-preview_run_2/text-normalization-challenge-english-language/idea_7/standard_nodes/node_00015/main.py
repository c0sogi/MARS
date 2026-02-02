import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import warnings

# Suppress warnings and progress bars where possible for clean output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from library.config import Config
from library.trainer import train_router_pipeline, train_generator_pipeline
from library.inference import HybridPredictor
from library.data_utils import process_router_data, get_router_dataloader
from library.rule_based_norm import apply_rule
from library.router_model import TokenClassifier
from library.generator_model import Seq2SeqNormalizer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from library.trainer import InferenceDataset, collate_inference


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    Config.set_seed(42)
    device = Config.DEVICE

    # Fast baseline settings to ensure execution within 2 hours
    # Router: Train on approx 100,000 sentences (approx 1.5M tokens)
    # Generator: Train on approx 500,000 raw rows (filtered down to unstructured tokens)
    ROUTER_TRAIN_SAMPLES = 100000
    GENERATOR_TRAIN_SAMPLES = 500000

    ROUTER_EPOCHS = 1
    GEN_EPOCHS = 2

    print(f"Running on device: {device}")

    # ==========================================
    # 2. Training Phase
    # ==========================================
    print("\n=== Starting Training Phase ===")

    # Train Router
    print("Training Router...")
    train_router_pipeline(
        epochs=ROUTER_EPOCHS,
        batch_size=32,
        debug_sample_size=ROUTER_TRAIN_SAMPLES,
        load_cached_data=False,  # Force reprocessing to apply sampling
    )

    # Train Generator
    print("Training Generator...")
    train_generator_pipeline(
        epochs=GEN_EPOCHS,
        batch_size=32,
        debug_sample_size=GENERATOR_TRAIN_SAMPLES,
        load_cached_data=False,
    )

    # ==========================================
    # 3. Validation Phase
    # ==========================================
    print("\n=== Starting Validation Phase ===")

    # Reset Config to use full dataset for validation
    Config.DEBUG_SAMPLE_SIZE = None

    # Load raw validation data for ground truth
    print("Loading full validation dataset...")
    val_df_raw = pd.read_csv(Config.VAL_DATA_PATH, keep_default_na=False)

    # Initialize Predictor (loads the models we just trained)
    predictor = HybridPredictor()

    # --- 3.1 Router Inference on Validation Set ---
    print("Running Router on Validation Set...")
    # Force reload of validation data (full set)
    val_loader = get_router_dataloader(split="val", load_cached_data=False)

    # We need to map batch predictions back to the flat dataframe structure
    # Load the grouped dataframe to iterate in sync with the loader
    df_val_grouped = process_router_data(split="val", load_cached_data=True)
    df_iter = df_val_grouped.itertuples(index=False)

    all_token_ids = []
    all_tokens = []
    all_pred_classes = []

    router_model = predictor.router_model
    router_tokenizer = predictor.router_tokenizer

    with torch.no_grad():
        # Disable tqdm for cleaner log output
        for batch in tqdm(val_loader, desc="Router Val", disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = router_model(input_ids, attention_mask).logits
            preds = torch.argmax(logits, dim=2).cpu().numpy()

            batch_len = input_ids.size(0)

            for i in range(batch_len):
                try:
                    row = next(df_iter)
                except StopIteration:
                    break

                raw_tokens = row.tokens
                row_ids = row.token_ids

                # Re-tokenize to align predictions
                encoding = router_tokenizer(
                    raw_tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=Config.ROUTER_MAX_LEN,
                    return_attention_mask=False,
                )
                word_ids = encoding.word_ids()

                sentence_preds = preds[i]
                aligned_preds = []
                prev_word_idx = None

                for j, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        continue
                    if word_idx != prev_word_idx:
                        if word_idx < len(raw_tokens):
                            class_id = sentence_preds[j]
                            aligned_preds.append(Config.ID2CLASS[class_id])
                        prev_word_idx = word_idx

                # Fallback for truncation
                if len(aligned_preds) < len(raw_tokens):
                    aligned_preds.extend(
                        ["PLAIN"] * (len(raw_tokens) - len(aligned_preds))
                    )

                all_token_ids.extend(row_ids)
                all_tokens.extend(raw_tokens)
                all_pred_classes.extend(aligned_preds)

    # --- 3.2 Hybrid Execution Logic ---
    print("Executing Hybrid Logic...")

    final_preds_map = {}
    gen_inputs = []
    gen_indices = []

    structured_set = Config.STRUCTURED_CLASSES
    unstructured_set = Config.UNSTRUCTURED_CLASSES

    for tid, token, cls in zip(all_token_ids, all_tokens, all_pred_classes):
        if cls == "PLAIN" or cls == "PUNCT":
            final_preds_map[tid] = token
        elif cls in structured_set:
            final_preds_map[tid] = apply_rule(token, cls)
        elif cls in unstructured_set:
            gen_inputs.append(f"[{cls}] {token}")
            gen_indices.append(tid)
        else:
            final_preds_map[tid] = token

    # --- 3.3 Generator Inference on Validation Set ---
    if gen_inputs:
        print(f"Running Generator on {len(gen_inputs)} validation tokens...")
        gen_dataset = InferenceDataset(
            gen_inputs, predictor.generator_tokenizer, max_len=Config.GEN_MAX_INPUT_LEN
        )
        gen_loader = DataLoader(
            gen_dataset,
            batch_size=Config.GEN_VAL_BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_inference,
        )

        gen_outputs = []
        generator_model = predictor.generator_model

        with torch.no_grad():
            for batch in tqdm(gen_loader, desc="Generator Val", disable=True):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                generated_ids = generator_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=Config.GEN_MAX_TARGET_LEN,
                )

                decoded = predictor.generator_tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                gen_outputs.extend(decoded)

        for tid, out_text in zip(gen_indices, gen_outputs):
            final_preds_map[tid] = out_text

    # ==========================================
    # 4. Metric Calculation
    # ==========================================
    print("Calculating Metrics...")

    # Map predictions to the raw validation dataframe
    val_df_raw["pred"] = val_df_raw["id"].map(final_preds_map)

    # Fill any missing predictions with raw text (safety fallback)
    val_df_raw["pred"] = val_df_raw["pred"].fillna(val_df_raw["before"])

    # Calculate exact match accuracy
    correct_count = (val_df_raw["pred"] == val_df_raw["after"]).sum()
    total_count = len(val_df_raw)
    accuracy = correct_count / total_count

    print(f"Final Validation Metric: {accuracy:.16f}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")

    val_df_raw["is_error"] = (val_df_raw["pred"] != val_df_raw["after"]).astype(int)
    val_df_raw["token_len"] = val_df_raw["before"].str.len()

    # Correlation between error and token length
    corr = val_df_raw["is_error"].corr(val_df_raw["token_len"])
    print(f"Correlation (Error vs Token Length): {corr:.6f}")

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 0.973229717044087

    if accuracy > THRESHOLD:
        print(
            f"\nValidation metric {accuracy:.6f} > {THRESHOLD}. Generating Submission..."
        )
        # Run inference on Test Set
        # Note: predictor.predict handles loading test data and saving to submission.csv
        predictor.predict(load_cached_data=True)
    else:
        print(f"\nValidation metric {accuracy:.6f} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
