import os
import gc
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, jaccard, AverageMeter
from library.data import get_processed_data, TweetDataset, SmartBatchingCollate
from library.model import TweetModel
from library.engine import train_fn, loss_fn

# ====================================================
# Configuration Overrides for Fast Baseline
# ====================================================
# Limit training parameters to ensure completion within the 2-hour limit
Config.EPOCHS = 2
Config.N_FOLDS = 5
TRAIN_SAMPLE_SIZE = 5000  # Subsample size for speed
Config.OUTPUT_DIR = "./working/output"
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


def run_training():
    """
    Executes the 5-fold Stratified Cross-Validation training loop.
    Returns a list of dictionaries containing model configuration and saved paths.
    """
    seed_everything(Config.SEED)

    # 1. Load and Subsample Data
    print("Loading and subsampling training data...")
    full_train_df = pd.read_csv("./metadata/train.csv")
    # Ensure clean data
    full_train_df = full_train_df.dropna(
        subset=["text", "selected_text", "sentiment"]
    ).reset_index(drop=True)

    # Stratified Subsampling to reduce runtime
    if len(full_train_df) > TRAIN_SAMPLE_SIZE:
        splitter = StratifiedKFold(
            n_splits=int(len(full_train_df) / TRAIN_SAMPLE_SIZE),
            shuffle=True,
            random_state=Config.SEED,
        )
        # Take the first fold as the subsample
        for _, subset_idx in splitter.split(full_train_df, full_train_df["sentiment"]):
            train_df = full_train_df.iloc[subset_idx].reset_index(drop=True)
            break
        # Fallback safety
        if len(train_df) > TRAIN_SAMPLE_SIZE * 1.5:
            train_df = train_df.head(TRAIN_SAMPLE_SIZE)
    else:
        train_df = full_train_df

    print(f"Training on {len(train_df)} samples.")

    # 2. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    trained_models = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["sentiment"])
    ):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Create DataFrames for this fold
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # Sort train data by length for Smart Batching (minimizes padding)
        fold_train_df["text_len"] = fold_train_df["text"].astype(str).apply(len)
        fold_train_df = fold_train_df.sort_values("text_len").reset_index(drop=True)
        fold_train_df = fold_train_df.drop(columns=["text_len"])

        # Iterate over both architectures (DeBERTa, RoBERTa)
        for model_cfg in Config.MODEL_CONFIGS:
            model_name = model_cfg["model_name"]
            save_name = f"{model_cfg['save_name']}_fold{fold}"
            print(f"Training {model_name}...")

            # Initialize Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            # Process Data
            # Note: We use unique cache keys per fold to avoid collisions
            train_data = get_processed_data(
                fold_train_df,
                tokenizer,
                Config.MAX_LEN,
                model_name,
                f"train_f{fold}",
                load_cached_data=True,
            )

            train_dataset = TweetDataset(train_data)
            collate_fn = SmartBatchingCollate(pad_token_id=tokenizer.pad_token_id)

            # Create DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=model_cfg["batch_size"],
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                collate_fn=collate_fn,
                pin_memory=True,
            )

            # Initialize Model
            device = Config.DEVICE
            model = TweetModel(model_name).to(device)

            # Optimization
            optimizer = AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            num_train_steps = int(len(train_loader) * Config.EPOCHS)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
                num_training_steps=num_train_steps,
            )

            # Training Loop
            for epoch in range(Config.EPOCHS):
                avg_loss = train_fn(train_loader, model, optimizer, device, scheduler)
                print(f"  Epoch {epoch+1} Loss: {avg_loss:.4f}")

            # Save Model State
            save_path = os.path.join(Config.OUTPUT_DIR, f"{save_name}.pth")
            torch.save(model.state_dict(), save_path)

            # Cleanup to free GPU memory
            del model, optimizer, scheduler, train_loader, train_dataset
            torch.cuda.empty_cache()
            gc.collect()

            trained_models.append(
                {"config": model_cfg, "path": save_path, "fold": fold}
            )

    return trained_models


def inference_ensemble(models_info, df_path, stage="test"):
    """
    Performs ensemble inference using Character-Level Probability Aggregation.
    This handles the tokenizer diversity by mapping token probs to character offsets.
    """
    print(f"\nRunning Inference on {stage} set...")
    df = pd.read_csv(df_path)
    df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

    # Initialize aggregated character probabilities
    # Max char length in dataset is ~141. 200 is a safe upper bound.
    MAX_CHAR_LEN = 200
    N_SAMPLES = len(df)

    agg_start_probs = np.zeros((N_SAMPLES, MAX_CHAR_LEN), dtype=np.float32)
    agg_end_probs = np.zeros((N_SAMPLES, MAX_CHAR_LEN), dtype=np.float32)

    device = Config.DEVICE

    # Iterate through all trained models
    for info in models_info:
        model_cfg = info["config"]
        model_path = info["path"]
        model_name = model_cfg["model_name"]

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Process Data
        data = get_processed_data(
            df, tokenizer, Config.MAX_LEN, model_name, stage, load_cached_data=True
        )
        dataset = TweetDataset(data)
        collate_fn = SmartBatchingCollate(pad_token_id=tokenizer.pad_token_id)

        # Double batch size for inference speed
        loader = DataLoader(
            dataset,
            batch_size=model_cfg["batch_size"] * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

        # Load Model
        model = TweetModel(model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Inference Loop
        sample_idx = 0
        with torch.no_grad():
            for batch in tqdm(
                loader, desc=f"Infer {os.path.basename(model_path)}", leave=False
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                offsets = batch["offsets"].cpu().numpy()

                # Get Logits
                start_logits, end_logits = model(
                    input_ids, attention_mask, token_type_ids
                )

                # Convert to Probabilities
                start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
                end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

                # Aggregate to Character Level
                batch_size = input_ids.size(0)

                for i in range(batch_size):
                    global_i = sample_idx + i
                    sample_offsets = offsets[i]
                    sample_start_probs = start_probs[i]
                    sample_end_probs = end_probs[i]

                    # Map tokens to characters using offsets
                    for token_j, (start_char, end_char) in enumerate(sample_offsets):
                        # Skip special tokens (usually 0,0) unless it's the very first char
                        if start_char == 0 and end_char == 0 and token_j != 0:
                            continue

                        # Accumulate probabilities
                        if start_char < MAX_CHAR_LEN:
                            agg_start_probs[global_i, start_char] += sample_start_probs[
                                token_j
                            ]
                        # End token points to the end of the span, so the character index is end_char - 1
                        if (end_char - 1) < MAX_CHAR_LEN and (end_char - 1) >= 0:
                            agg_end_probs[global_i, end_char - 1] += sample_end_probs[
                                token_j
                            ]

                sample_idx += batch_size

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Decode Predictions
    predictions = []
    texts = df["text"].values
    sentiments = df["sentiment"].values

    for i in range(N_SAMPLES):
        text = str(texts[i])
        sentiment = sentiments[i]

        # Neutral Heuristic: Predict full text
        if sentiment == "neutral":
            predictions.append(text)
        else:
            # Find best start/end indices in the aggregated character probability maps
            start_char_idx = np.argmax(agg_start_probs[i])
            end_char_idx = np.argmax(agg_end_probs[i])

            # Constraint: End must be after Start
            if end_char_idx < start_char_idx:
                end_char_idx = start_char_idx

            # Extract text (end_char_idx is inclusive, so +1 for slice)
            # Ensure indices are within bounds of the text
            if start_char_idx >= len(text):
                start_char_idx = 0
            if end_char_idx >= len(text):
                end_char_idx = len(text) - 1

            pred = text[start_char_idx : end_char_idx + 1]
            predictions.append(pred)

    return predictions


def main():
    # 1. Train Models
    trained_models = run_training()

    # 2. Validate on Hold-out Set
    val_preds = inference_ensemble(trained_models, "./metadata/val.csv", stage="val")
    val_df = pd.read_csv("./metadata/val.csv")
    val_df = val_df.dropna(subset=["text", "sentiment", "selected_text"])

    # Compute Jaccard Score
    scores = []
    for i in range(len(val_df)):
        score = jaccard(val_preds[i], val_df["selected_text"].iloc[i])
        scores.append(score)

    final_metric = np.mean(scores)
    # Required Output Format
    print(f"Final Validation Metric: {final_metric}")

    # 3. Failure Analysis
    print("\nFailure Analysis:")
    val_df["jaccard"] = scores
    val_df["error"] = 1.0 - val_df["jaccard"]
    val_df["text_len"] = val_df["text"].astype(str).apply(len)

    # Correlation Analysis
    corr_len = val_df["error"].corr(val_df["text_len"])
    print(f"Correlation (Error vs Text Length): {corr_len:.4f}")

    # Sentiment Breakdown
    print("Mean Jaccard by Sentiment:")
    print(val_df.groupby("sentiment")["jaccard"].mean())

    # 4. Generate Submission
    if final_metric > 0.7205:
        print("\nGenerating Submission...")
        test_preds = inference_ensemble(
            trained_models, "./metadata/test.csv", stage="test"
        )

        sub_df = pd.read_csv("./metadata/test.csv")
        sub_df["selected_text"] = test_preds

        # Save to CSV
        os.makedirs("./submission", exist_ok=True)
        sub_df[["textID", "selected_text"]].to_csv(
            "./submission/submission.csv", index=False
        )
        print("Submission saved to ./submission/submission.csv")
    else:
        print(f"\nMetric {final_metric} <= 0.7205. Skipping submission.")


if __name__ == "__main__":
    main()
