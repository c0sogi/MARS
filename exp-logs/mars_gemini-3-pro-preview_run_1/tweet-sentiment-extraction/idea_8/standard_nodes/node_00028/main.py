import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, logging

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import load_and_cache_data
from library.model import TweetModel
from library.engine import train_fn, eval_fn, predict_test_set, decode_batch


def run():
    # Suppress transformer warnings for cleaner output
    logging.set_verbosity_error()

    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Optimization: 5 Epochs for maximum convergence
    Config.EPOCHS = 5

    print(f"Starting pipeline on device: {device}")

    # 2. Data Loading
    # Load training data (neutrals filtered out based on Config.FILTER_NEUTRAL)
    print("Loading Training Data...")
    train_dataset = load_and_cache_data("train", load_cached_data=True)

    # Load validation data (neutrals included for full evaluation)
    print("Loading Validation Data...")
    val_dataset = load_and_cache_data("val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing TweetModel...")
    model = TweetModel()
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = int(len(train_dataset) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 5. Training Loop
    print("Starting Training...")
    best_jaccard = -1.0

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_loss, val_jaccard = eval_fn(val_loader, model, device)

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Saved Best Model! Jaccard: {best_jaccard:.4f}")

    # 6. Final Validation & Failure Analysis
    print("\n--- Final Evaluation & Failure Analysis ---")

    # Load best model for analysis
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)
    model.eval()

    # Collect detailed predictions for analysis
    val_ids = []
    val_texts = []
    val_selected = []
    val_sentiments = []
    val_preds = []
    val_jaccards = []

    with torch.no_grad():
        for d in val_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)

            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            start_logits_np = start_logits.cpu().numpy()
            end_logits_np = end_logits.cpu().numpy()
            offsets_np = d["offsets"].numpy()
            texts = d["text"]
            sentiments = d["sentiment"]
            selected_texts = d["selected_text"]
            ids = d["textID"]

            preds = decode_batch(
                start_logits_np, end_logits_np, offsets_np, texts, sentiments
            )

            for i in range(len(preds)):
                score = jaccard(preds[i], selected_texts[i])
                val_ids.append(ids[i])
                val_texts.append(texts[i])
                val_selected.append(selected_texts[i])
                val_sentiments.append(sentiments[i])
                val_preds.append(preds[i])
                val_jaccards.append(score)

    final_metric = np.mean(val_jaccards)
    # Print the required metric format
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    df_analysis = pd.DataFrame(
        {"text": val_texts, "sentiment": val_sentiments, "jaccard": val_jaccards}
    )

    # Calculate Error Magnitude
    df_analysis["error"] = 1.0 - df_analysis["jaccard"]

    # Feature Engineering for Correlation
    df_analysis["text_len"] = df_analysis["text"].apply(len)
    sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
    df_analysis["sentiment_enc"] = df_analysis["sentiment"].map(sentiment_map)

    # Compute Correlations
    corr_len = df_analysis["text_len"].corr(df_analysis["error"])
    corr_sent = df_analysis["sentiment_enc"].corr(df_analysis["error"])

    print(f"Correlation (Text Length vs Error): {corr_len}")
    print(f"Correlation (Sentiment vs Error): {corr_sent}")

    # 7. Submission Generation
    THRESHOLD = 0.7043342108129372

    if final_metric > THRESHOLD:
        print("\nMetric threshold met. Generating submission...")
        test_dataset = load_and_cache_data("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        predict_test_set(test_loader, model, device)
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
