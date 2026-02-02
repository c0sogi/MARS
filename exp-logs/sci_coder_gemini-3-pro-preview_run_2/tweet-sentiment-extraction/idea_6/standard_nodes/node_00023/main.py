import os
import sys
import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.dataset import get_data
from library.model import TweetModel
from library.loss import HybridLoss
from library.engine import train_fn
from library.utils import seed_everything, jaccard


def run():
    # 1. Setup and Configuration Overrides for Fast Baseline
    seed_everything(Config.seed)
    device = Config.device

    # Override Config for speed constraints (ensure execution < 2 hours)
    Config.epochs = 2

    # 2. Data Loading
    print("Loading data...")
    # get_data loads metadata/train.csv, metadata/val.csv, metadata/test.csv
    # train_dataset -> metadata/train.csv
    # holdout_val_dataset -> metadata/val.csv (Hold-out set for final metric)
    # test_dataset -> metadata/test.csv
    train_dataset, holdout_val_dataset, test_dataset = get_data(load_cached_data=True)

    # Prepare for Stratified K-Fold on the training set
    sentiments = train_dataset.sentiments
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    model_paths = []

    # 3. Training Loop (5 Folds)
    print(f"Starting training: {Config.n_folds} folds, {Config.epochs} epochs each.")

    # We split the training dataset into folds.
    # Note: We train on 'train_dataset' and evaluate on 'holdout_val_dataset' at the very end.
    for fold, (train_idx, _) in enumerate(
        skf.split(np.zeros(len(sentiments)), sentiments)
    ):
        print(f"\n--- Fold {fold + 1}/{Config.n_folds} ---")

        # Subsample training data to 50% to meet time constraints while maintaining diversity
        np.random.shuffle(train_idx)
        subset_size = int(len(train_idx) * 0.5)
        train_idx_sub = train_idx[:subset_size]

        train_sub = Subset(train_dataset, train_idx_sub)

        train_loader = DataLoader(
            train_sub,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        model = TweetModel()
        model.to(device)

        # Optimizer and Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )
        num_train_steps = int(len(train_loader) * Config.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        criterion = HybridLoss()

        # Training Epochs
        for epoch in range(Config.epochs):
            avg_loss = train_fn(
                train_loader, model, optimizer, device, scheduler, criterion
            )
            print(f"  Epoch {epoch + 1} Loss: {avg_loss:.4f}")

        # Save Model
        model_path = os.path.join(Config.MODEL_OUTPUT_DIR, f"model_fold_{fold}.pth")
        torch.save(model.state_dict(), model_path)
        model_paths.append(model_path)

        # Cleanup to save memory
        del model, optimizer, scheduler, train_loader, train_sub
        torch.cuda.empty_cache()

    # 4. Validation on Hold-out Set (Ensemble)
    print("\n--- Validating on Hold-out Set ---")
    holdout_loader = DataLoader(
        holdout_val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Load all models for ensemble
    models = []
    for path in model_paths:
        m = TweetModel()
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        m.eval()
        models.append(m)

    all_jaccards = []
    fa_data = []  # Failure analysis data

    with torch.no_grad():
        for data in holdout_loader:
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)

            orig_tweets = data["orig_tweet"]
            sentiments = data["sentiment"]
            orig_selected = data["orig_selected"]
            offsets = data["offsets"].numpy()

            # Ensemble Prediction: Sum logits
            start_logits_sum = None
            end_logits_sum = None

            for m in models:
                s, e = m(ids, mask, token_type_ids)
                if start_logits_sum is None:
                    start_logits_sum = s
                    end_logits_sum = e
                else:
                    start_logits_sum += s
                    end_logits_sum += e

            # Average logits
            start_logits_avg = (start_logits_sum / len(models)).cpu().numpy()
            end_logits_avg = (end_logits_sum / len(models)).cpu().numpy()

            # Decoding
            for i in range(len(ids)):
                tweet = orig_tweets[i]
                sentiment = sentiments[i]
                target = orig_selected[i]
                offset = offsets[i]

                s_log = start_logits_avg[i]
                e_log = end_logits_avg[i]

                if Config.neutral_heuristic and sentiment == "neutral":
                    pred = tweet
                else:
                    # Maximizing start + end logits
                    scores = s_log[:, np.newaxis] + e_log[np.newaxis, :]
                    upper_tri_mask = np.triu(np.ones_like(scores), k=0)
                    scores = np.where(upper_tri_mask == 1, scores, -np.inf)
                    max_idx = np.argmax(scores)
                    idx_start, idx_end = np.unravel_index(max_idx, scores.shape)

                    # Map back to chars using offsets
                    if idx_start < len(offset) and idx_end < len(offset):
                        char_start = offset[idx_start][0]
                        char_end = offset[idx_end][1]
                        pred = tweet[char_start:char_end]
                    else:
                        pred = tweet

                score = jaccard(pred, target)
                all_jaccards.append(score)

                fa_data.append(
                    {
                        "error": 1.0 - score,
                        "sentiment": sentiment,
                        "text_len": len(tweet.split()),
                    }
                )

    final_metric = np.mean(all_jaccards)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_fa = pd.DataFrame(fa_data)
    corr_len = df_fa["error"].corr(df_fa["text_len"])
    print(f"Correlation (Error vs Input Length): {corr_len:.4f}")

    print("Mean Error by Sentiment:")
    print(df_fa.groupby("sentiment")["error"].mean())

    # 6. Submission
    if final_metric > 0.7205:
        print("\n--- Generating Submission ---")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        predictions = []

        with torch.no_grad():
            for data in test_loader:
                ids = data["ids"].to(device)
                mask = data["mask"].to(device)
                token_type_ids = data["token_type_ids"].to(device)

                orig_tweets = data["orig_tweet"]
                sentiments = data["sentiment"]
                offsets = data["offsets"].numpy()

                # Ensemble Prediction
                start_logits_sum = None
                end_logits_sum = None

                for m in models:
                    s, e = m(ids, mask, token_type_ids)
                    if start_logits_sum is None:
                        start_logits_sum = s
                        end_logits_sum = e
                    else:
                        start_logits_sum += s
                        end_logits_sum += e

                start_logits_avg = (start_logits_sum / len(models)).cpu().numpy()
                end_logits_avg = (end_logits_sum / len(models)).cpu().numpy()

                for i in range(len(ids)):
                    tweet = orig_tweets[i]
                    sentiment = sentiments[i]
                    offset = offsets[i]

                    s_log = start_logits_avg[i]
                    e_log = end_logits_avg[i]

                    if Config.neutral_heuristic and sentiment == "neutral":
                        pred = tweet
                    else:
                        scores = s_log[:, np.newaxis] + e_log[np.newaxis, :]
                        upper_tri_mask = np.triu(np.ones_like(scores), k=0)
                        scores = np.where(upper_tri_mask == 1, scores, -np.inf)
                        max_idx = np.argmax(scores)
                        idx_start, idx_end = np.unravel_index(max_idx, scores.shape)

                        if idx_start < len(offset) and idx_end < len(offset):
                            char_start = offset[idx_start][0]
                            char_end = offset[idx_end][1]
                            pred = tweet[char_start:char_end]
                        else:
                            pred = tweet

                    predictions.append(pred)

        # Create Submission File
        df_test = pd.read_csv(Config.TEST_META)
        submission = pd.DataFrame(
            {"textID": df_test["textID"], "selected_text": predictions}
        )

        # Save with quoting (QUOTE_NONNUMERIC ensures strings are quoted)
        submission.to_csv(
            Config.SUBMISSION_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC
        )
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(f"Validation metric {final_metric:.4f} did not meet threshold 0.7205.")


if __name__ == "__main__":
    run()
