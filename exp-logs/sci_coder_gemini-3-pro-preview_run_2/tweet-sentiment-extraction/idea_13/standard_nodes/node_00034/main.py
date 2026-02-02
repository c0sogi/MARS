import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import get_loaders, TweetDataset, _find_target_indices
from library.model import TweetModel
from library.engine import train_fn, eval_fn, decode_prediction
from library.inference import predict_test

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# We override the default configuration to ensure the pipeline runs within the time limit.
Config.epochs = 1  # Train for 1 epoch per stage to ensure completion within 2 hours
Config.train_batch_size = 16  # Safe batch size for DeBERTa-large on A100
Config.valid_batch_size = 64
Config.output_dir = "./working/submission_run"  # Dedicated output directory
os.makedirs(Config.output_dir, exist_ok=True)


# ==========================================
# Helper Functions
# ==========================================


def get_holdout_loader(tokenizer):
    """
    Manually loads metadata/val.csv and prepares a DataLoader.
    This ensures we evaluate strictly on the hold-out set without data leakage
    from the concatenation logic in the library's process_data function.
    """
    print("Processing hold-out validation data from ./metadata/val.csv...")
    df = pd.read_csv(Config.val_path)
    # Ensure data integrity
    df.dropna(subset=["text", "sentiment", "selected_text"], inplace=True)

    size = len(df)
    input_ids = np.zeros((size, Config.max_len), dtype=np.int32)
    attention_masks = np.zeros((size, Config.max_len), dtype=np.int32)
    token_type_ids = np.zeros((size, Config.max_len), dtype=np.int32)
    offsets = np.zeros((size, Config.max_len, 2), dtype=np.int32)
    targets = np.zeros((size, 2), dtype=np.int32)

    for i, row in enumerate(df.itertuples()):
        text = str(row.text).strip()
        sentiment = str(row.sentiment).strip()
        selected_text = str(row.selected_text).strip()

        encoded = tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=Config.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        input_ids[i] = encoded["input_ids"]
        attention_masks[i] = encoded["attention_mask"]
        token_type_ids[i] = encoded["token_type_ids"]
        offsets[i] = encoded["offset_mapping"]

        # Compute targets for Jaccard evaluation
        start_idx, end_idx = _find_target_indices(
            text, selected_text, encoded["offset_mapping"], encoded.sequence_ids()
        )
        targets[i] = [start_idx, end_idx]

    dataset = TweetDataset(
        input_ids,
        attention_masks,
        token_type_ids,
        offsets,
        df["text"].values,
        df["sentiment"].values,
        df["selected_text"].values,
        targets,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return loader, df


# ==========================================
# Main Execution Pipeline
# ==========================================


def run():
    seed_everything(Config.seed)
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # ---------------------------------------------------------
    # Stage 1: Teacher Training & OOF Generation
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("STAGE 1: Teacher Training")
    print("=" * 40)

    # Dictionary to store soft labels: text -> (start_logits, end_logits)
    oof_logits = {}

    for fold in range(Config.n_folds):
        print(f"\n>>> Fold {fold} / {Config.n_folds}")
        train_loader, val_loader = get_loaders(fold, tokenizer)

        model = TweetModel()
        model.to(Config.device)

        optimizer = AdamW(
            model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
        )
        num_train_steps = len(train_loader) * Config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Train Loop
        for epoch in range(Config.epochs):
            train_fn(train_loader, model, optimizer, Config.device, scheduler=scheduler)

        # Generate OOF Soft Labels on the validation part of this fold
        # eval_fn returns logits_dict mapping text -> logits
        _, _, logits_dict = eval_fn(val_loader, model, Config.device)
        oof_logits.update(logits_dict)

        # Cleanup to save memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Stage 2: Student Training (Distillation)
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("STAGE 2: Student Training (Distillation)")
    print("=" * 40)

    for fold in range(Config.n_folds):
        print(f"\n>>> Fold {fold} / {Config.n_folds}")
        train_loader, val_loader = get_loaders(fold, tokenizer)

        model = TweetModel()
        model.to(Config.device)

        optimizer = AdamW(
            model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
        )
        num_train_steps = len(train_loader) * Config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        for epoch in range(Config.epochs):
            # Pass soft_labels_cache to enable DistillationLoss
            train_fn(
                train_loader,
                model,
                optimizer,
                Config.device,
                scheduler=scheduler,
                soft_labels_cache=oof_logits,
            )

        # Save Final Student Model
        save_path = os.path.join(Config.output_dir, f"model_fold_{fold}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Validation on Hold-out Set
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("VALIDATION & FAILURE ANALYSIS")
    print("=" * 40)

    val_loader, val_df = get_holdout_loader(tokenizer)

    # Ensemble Inference
    print("Running Ensemble Inference on Hold-out Set...")

    num_samples = len(val_df)
    final_start_logits = np.zeros((num_samples, Config.max_len))
    final_end_logits = np.zeros((num_samples, Config.max_len))

    # Accumulate logits from all folds
    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.output_dir, f"model_fold_{fold}.pth")
        model = TweetModel()
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)
        model.eval()

        fold_start = []
        fold_end = []

        with torch.no_grad():
            for data in val_loader:
                input_ids = data["input_ids"].to(Config.device)
                attention_mask = data["attention_mask"].to(Config.device)
                token_type_ids = data["token_type_ids"].to(Config.device)

                s, e = model(input_ids, attention_mask, token_type_ids)
                fold_start.append(s.cpu().numpy())
                fold_end.append(e.cpu().numpy())

        final_start_logits += np.concatenate(fold_start, axis=0)
        final_end_logits += np.concatenate(fold_end, axis=0)

        del model
        torch.cuda.empty_cache()

    # Decode Predictions
    start_idxs = np.argmax(final_start_logits, axis=1)
    end_idxs = np.argmax(final_end_logits, axis=1)

    predictions = []
    jaccard_scores = []

    offsets = val_loader.dataset.offsets
    texts = val_df["text"].values
    sentiments = val_df["sentiment"].values
    selected_texts = val_df["selected_text"].values

    for i in range(num_samples):
        pred = decode_prediction(
            start_idxs[i], end_idxs[i], str(texts[i]), offsets[i], str(sentiments[i])
        )
        predictions.append(pred)
        jaccard_scores.append(jaccard(str(selected_texts[i]), pred))

    final_metric = np.mean(jaccard_scores)
    # Print metric with full precision as required
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    val_df["jaccard"] = jaccard_scores
    val_df["error"] = 1.0 - val_df["jaccard"]
    val_df["text_len"] = val_df["text"].apply(len)

    # Calculate correlation using numpy
    corr = np.corrcoef(val_df["error"], val_df["text_len"])[0, 1]
    print(f"Correlation between Error and Text Length: {corr}")

    # ---------------------------------------------------------
    # Submission
    # ---------------------------------------------------------
    if final_metric > 0.7205:
        print("\n" + "=" * 40)
        print("GENERATING SUBMISSION")
        print("=" * 40)

        # Clear any existing cache to ensure test set is processed correctly
        cache_files = [
            os.path.join(Config.output_dir, f"cached_test_{Config.max_len}.npz"),
            os.path.join(Config.output_dir, f"cached_test_{Config.max_len}.parquet"),
        ]
        for f in cache_files:
            if os.path.exists(f):
                os.remove(f)

        # Generate submission.csv
        predict_test(load_cached_data=False)


if __name__ == "__main__":
    run()
