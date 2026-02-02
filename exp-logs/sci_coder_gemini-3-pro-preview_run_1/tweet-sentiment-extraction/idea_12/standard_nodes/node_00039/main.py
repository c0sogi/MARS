import pandas as pd
import numpy as np
import torch
import os
import sys
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, normalize_text, jaccard
from library.data import get_data_loaders, get_test_loader
from library.model import TweetModel
from library.engine import train_fn, eval_fn, inference_fn, get_optimizer_params


def run():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Fast baseline overrides
    Config.EPOCHS = 2  # Reduce epochs for speed
    Config.DEBUG = False

    print(f"Starting execution on device: {device}")

    # 2. Data Loading (Fold 0)
    # Load training and validation loaders for the first fold
    train_loader, val_loader = get_data_loaders(fold=0, load_cached_data=True)

    # 3. Model Initialization
    model = TweetModel()
    model.to(device)

    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params, lr=Config.LR_MAX, eps=Config.EPS)

    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    num_warmup_steps = int(num_train_steps * Config.NUM_WARMUP_STEPS_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_train_steps
    )

    # 4. Training Loop
    best_jaccard = 0.0

    # We need a validation dataframe subset for the eval_fn during training
    # Reconstruct the fold split to get the non-neutral validation dataframe
    full_train_df = pd.read_csv(Config.TRAIN_META)
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(full_train_df, full_train_df["sentiment"]))
    _, val_idx = splits[0]

    # Filter for non-neutrals for the training-loop validation (matching val_loader)
    if Config.FILTER_NEUTRAL:
        is_not_neutral = full_train_df["sentiment"] != "neutral"
        val_mask = is_not_neutral.iloc[val_idx].values
        val_idx_active = val_idx[val_mask]
        val_df_active = full_train_df.iloc[val_idx_active].reset_index(drop=True)
    else:
        val_df_active = full_train_df.iloc[val_idx].reset_index(drop=True)

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_loss, val_jaccard = eval_fn(val_loader, model, device, val_df_active)

        # print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Jaccard: {val_jaccard:.4f}")

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # 5. Final Validation Assessment (Entire Hold-out Set)
    print("Performing Final Validation...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get the FULL validation set (including neutrals)
    val_df_full = full_train_df.iloc[val_idx].reset_index(drop=True)

    # Split into active (non-neutral) and neutral for separate processing
    val_df_active_full = val_df_full[val_df_full["sentiment"] != "neutral"].reset_index(
        drop=True
    )
    val_df_neutral_full = val_df_full[
        val_df_full["sentiment"] == "neutral"
    ].reset_index(drop=True)

    final_scores = []
    text_lens = []
    sentiments_encoded = []  # 0: neg, 1: neu, 2: pos

    # A. Process Non-Neutrals (Active)
    # We reuse the val_loader which corresponds exactly to val_df_active
    # inference_fn returns quoted strings, we strip quotes for metric calc
    active_preds_quoted = inference_fn(val_loader, model, device, val_df_active_full)
    active_preds = [p.strip('"') for p in active_preds_quoted]

    for i, pred in enumerate(active_preds):
        target = normalize_text(str(val_df_active_full.iloc[i]["selected_text"]))
        score = jaccard(pred, target)
        final_scores.append(score)

        # Features for failure analysis
        txt = str(val_df_active_full.iloc[i]["text"])
        text_lens.append(len(txt))
        sent = val_df_active_full.iloc[i]["sentiment"]
        sentiments_encoded.append(0 if sent == "negative" else 2)

    # B. Process Neutrals (Identity Prediction)
    for i, row in val_df_neutral_full.iterrows():
        text = normalize_text(str(row["text"]))
        target = normalize_text(str(row["selected_text"]))
        # Neutral strategy: predict full text
        score = jaccard(text, target)
        final_scores.append(score)

        # Features
        text_lens.append(len(str(row["text"])))
        sentiments_encoded.append(1)

    # Compute Final Metric
    final_metric = np.mean(final_scores)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    errors = 1.0 - np.array(final_scores)
    text_lens = np.array(text_lens)
    sentiments_encoded = np.array(sentiments_encoded)

    # Correlations
    if len(errors) > 1:
        corr_len = np.corrcoef(errors, text_lens)[0, 1]
        corr_sent = np.corrcoef(errors, sentiments_encoded)[0, 1]
    else:
        corr_len = 0.0
        corr_sent = 0.0

    print(f"Correlation (Error vs Text Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Sentiment): {corr_sent:.4f}")

    # 7. Submission Generation
    if final_metric > 0.7093:
        print("Validation metric threshold met. Generating submission...")
        test_loader, test_df = get_test_loader(load_cached_data=True)

        # inference_fn handles the neutral strategy internally for the test set
        predictions = inference_fn(test_loader, model, device, test_df)

        submission_df = pd.DataFrame(
            {"textID": test_df["textID"], "selected_text": predictions}
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_metric:.4f} did not meet threshold 0.7093. Submission skipped."
        )


if __name__ == "__main__":
    run()
