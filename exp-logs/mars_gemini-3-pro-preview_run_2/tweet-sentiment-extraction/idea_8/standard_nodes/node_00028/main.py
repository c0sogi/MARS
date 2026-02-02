import os
import gc
import sys
import csv
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import seed_everything, AverageMeter, jaccard
from library.data import (
    process_data,
    TweetDataset,
    SmartBatchSampler,
    SmartBatchingCollate,
)
from library.model import TweetModel
from library.engine import train_fn, eval_fn


def run_fold(fold, train_idx, val_idx, df_full, data_full, tokenizer, device):
    print(f"\n{'='*20} Fold {fold+1}/{Config.NUM_FOLDS} {'='*20}")

    # 1. Prepare Data for this Fold
    # Slice the arrays from data_full
    # We filter for keys that are numpy arrays to avoid slicing non-array items if any
    train_data = {
        k: v[train_idx] for k, v in data_full.items() if isinstance(v, np.ndarray)
    }
    val_data = {
        k: v[val_idx] for k, v in data_full.items() if isinstance(v, np.ndarray)
    }

    # Create Datasets
    train_dataset = TweetDataset(df_full.iloc[train_idx], train_data)
    val_dataset = TweetDataset(df_full.iloc[val_idx], val_data)

    # Create Loaders
    collate_fn = SmartBatchingCollate(tokenizer)

    # SmartBatchSampler for training to speed up processing
    train_sampler = SmartBatchSampler(
        train_dataset, Config.TRAIN_BATCH_SIZE, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model & Optimizer
    model = TweetModel()
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_train_steps = int(len(train_dataset) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 3. Training Loop
    best_jaccard = 0
    model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_jaccard = eval_fn(val_loader, model, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
        )

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), model_path)

    print(f"Best Jaccard for Fold {fold+1}: {best_jaccard:.4f}")

    # 4. Generate OOF Predictions
    # Load best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    oof_preds = []

    with torch.no_grad():
        for data in val_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)

            offsets = data["offsets"].cpu().numpy()
            texts = data["text"]
            sentiments = data["sentiment"]

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            for i in range(len(texts)):
                text = texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]

                if sentiment == "neutral":
                    pred_text = text
                else:
                    start_l = start_logits[i]
                    end_l = end_logits[i]

                    # Decoding
                    score_mat = start_l[:, None] + end_l[None, :]
                    upper_tri_mask = np.triu(np.ones_like(score_mat))
                    score_mat = np.where(upper_tri_mask == 1, score_mat, -np.inf)

                    best_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
                    idx_start, idx_end = best_idx

                    char_start = offset[idx_start][0]
                    char_end = offset[idx_end][1]

                    if char_start == 0 and char_end == 0:
                        pred_text = text
                    else:
                        pred_text = text[char_start:char_end]

                oof_preds.append(pred_text)

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return val_idx, oof_preds


def main():
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # --- Data Loading ---
    print("Loading Data...")
    train_df_part = pd.read_csv(Config.TRAIN_META_PATH)
    val_df_part = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    # Combine Train and Val to perform full 5-Fold CV
    df_full = pd.concat([train_df_part, val_df_part]).reset_index(drop=True)

    # Clean NaN values
    df_full.dropna(subset=["text", "selected_text", "sentiment"], inplace=True)
    df_full.reset_index(drop=True, inplace=True)

    # Handle NaNs in test text if any
    test_df["text"] = test_df["text"].fillna("")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Process Data
    print("Processing Training Data...")
    data_full = process_data(df_full, tokenizer, Config.MAX_LEN, is_test=False)

    print("Processing Test Data...")
    data_test = process_data(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    # --- Cross Validation Loop ---
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_indices = []
    oof_predictions = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_full, df_full["sentiment"])
    ):
        val_idx_fold, preds_fold = run_fold(
            fold, train_idx, val_idx, df_full, data_full, tokenizer, device
        )
        oof_indices.extend(val_idx_fold)
        oof_predictions.extend(preds_fold)

    # --- Validation Analysis ---
    print("\n--- Validation Analysis ---")

    # Map predictions back to original dataframe order
    pred_map = {idx: pred for idx, pred in zip(oof_indices, oof_predictions)}
    df_full["pred_selected_text"] = df_full.index.map(pred_map)

    # Calculate Jaccard
    df_full["jaccard"] = df_full.apply(
        lambda x: jaccard(x["selected_text"], x["pred_selected_text"]), axis=1
    )

    final_metric = df_full["jaccard"].mean()
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    df_full["error_magnitude"] = 1.0 - df_full["jaccard"]
    df_full["word_len"] = df_full["text"].apply(lambda x: len(str(x).split()))

    correlation = df_full["error_magnitude"].corr(df_full["word_len"])
    print(f"\nCorrelation (Error Magnitude vs Input Word Length): {correlation:.4f}")

    print("\nMean Jaccard by Sentiment:")
    print(df_full.groupby("sentiment")["jaccard"].mean())

    # --- Submission ---
    if final_metric > 0.7205:
        print("\nMetric threshold met. Generating submission...")

        # Prepare Test Loader
        test_dataset = TweetDataset(test_df, data_test)
        collate_fn = SmartBatchingCollate(tokenizer)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Accumulators for Ensemble Logits
        # Initialize to 0. Valid logits will be added.
        avg_start_logits = np.zeros((len(test_df), Config.MAX_LEN), dtype=np.float32)
        avg_end_logits = np.zeros((len(test_df), Config.MAX_LEN), dtype=np.float32)

        for fold in range(Config.NUM_FOLDS):
            print(f"Inference Fold {fold+1}...")
            model = TweetModel()
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            global_idx = 0

            with torch.no_grad():
                for data in test_loader:
                    input_ids = data["input_ids"].to(device)
                    attention_mask = data["attention_mask"].to(device)
                    token_type_ids = data["token_type_ids"].to(device)

                    start_logits, end_logits = model(
                        input_ids, attention_mask, token_type_ids
                    )

                    s_l = start_logits.cpu().numpy()
                    e_l = end_logits.cpu().numpy()

                    batch_size = s_l.shape[0]
                    seq_len = s_l.shape[1]

                    current_indices = range(global_idx, global_idx + batch_size)

                    # Accumulate logits.
                    # Note: s_l is trimmed to batch max_len. avg_start_logits is fixed MAX_LEN.
                    avg_start_logits[current_indices, :seq_len] += s_l
                    avg_end_logits[current_indices, :seq_len] += e_l

                    global_idx += batch_size

            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Decode Predictions
        print("Decoding predictions...")
        predictions = []

        offsets = data_test["offsets"]
        texts = test_df["text"].values
        sentiments = test_df["sentiment"].values
        ids = test_df["textID"].values
        attention_masks = data_test["attention_mask"]

        for i in range(len(test_df)):
            text = str(texts[i])
            sentiment = sentiments[i]
            offset = offsets[i]
            valid_len = np.sum(attention_masks[i])

            if sentiment == "neutral":
                pred_text = text
            else:
                # Slice logits to valid length to avoid padding
                start_l = avg_start_logits[i, :valid_len]
                end_l = avg_end_logits[i, :valid_len]

                score_mat = start_l[:, None] + end_l[None, :]
                upper_tri_mask = np.triu(np.ones_like(score_mat))
                score_mat = np.where(upper_tri_mask == 1, score_mat, -np.inf)

                best_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
                idx_start, idx_end = best_idx

                # Check bounds and padding
                if idx_start >= len(offset) or idx_end >= len(offset):
                    pred_text = text
                else:
                    char_start = offset[idx_start][0]
                    char_end = offset[idx_end][1]

                    # If predicted span is padding (0,0) or empty
                    if char_start == 0 and char_end == 0:
                        pred_text = text
                    else:
                        pred_text = text[char_start:char_end]

            predictions.append(pred_text)

        submission_df = pd.DataFrame({"textID": ids, "selected_text": predictions})

        # Save submission with quoting to handle special characters
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(
            Config.SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC
        )
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
