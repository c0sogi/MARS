import pandas as pd
import numpy as np
import torch
import os
import sys
import gc
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, normalize_text, jaccard
from library.data import get_data_loaders, get_eval_loader
from library.model import TweetModel
from library.engine import (
    train_fn,
    eval_fn,
    inference_fn_ensemble,
    get_optimizer_params,
)


def run():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Ensure Config is respected (remove overrides)
    print(f"Starting execution on device: {device}")
    print(f"Folds: {Config.N_FOLDS}, Epochs: {Config.EPOCHS}")

    # 2. Training Loop (All Folds)
    model_paths = []

    # Load full train df for splitting logic
    full_train_df = pd.read_csv(Config.TRAIN_META)
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(full_train_df, full_train_df["sentiment"]))

    for fold in range(Config.N_FOLDS):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Load Data
        train_loader, val_loader = get_data_loaders(fold=fold, load_cached_data=True)

        # Get Validation DF for this fold (active only for monitoring)
        _, val_idx = splits[fold]
        if Config.FILTER_NEUTRAL:
            is_not_neutral = full_train_df["sentiment"] != "neutral"
            val_mask = is_not_neutral.iloc[val_idx].values
            val_idx_active = val_idx[val_mask]
            val_df_active = full_train_df.iloc[val_idx_active].reset_index(drop=True)
        else:
            val_df_active = full_train_df.iloc[val_idx].reset_index(drop=True)

        # Initialize Model
        model = TweetModel()
        model.to(device)

        optimizer_params = get_optimizer_params(model)
        optimizer = torch.optim.AdamW(
            optimizer_params, lr=Config.LR_MAX, eps=Config.EPS
        )

        num_train_steps = int(len(train_loader) * Config.EPOCHS)
        num_warmup_steps = int(num_train_steps * Config.NUM_WARMUP_STEPS_RATIO)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_train_steps
        )

        # Train
        best_jaccard = 0.0
        fold_model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.bin")

        for epoch in range(Config.EPOCHS):
            train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
            val_loss, val_jaccard = eval_fn(val_loader, model, device, val_df_active)

            print(f"Epoch {epoch+1} | Val Jaccard: {val_jaccard:.5f}")

            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                torch.save(model.state_dict(), fold_model_path)

        print(f"Fold {fold+1} Best Jaccard: {best_jaccard:.5f}")
        model_paths.append(fold_model_path)

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Final Validation Assessment (Ensemble on Hold-out Set)
    print("\nPerforming Final Validation on Hold-out Set (VAL_META)...")

    # Load Validation Data (The true hold-out set)
    val_holdout_df = pd.read_csv(Config.VAL_META)
    val_loader = get_eval_loader(val_holdout_df, prefix="val_holdout")

    # Load All Models
    models = []
    for path in model_paths:
        m = TweetModel()
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        m.eval()
        models.append(m)

    # Ensemble Inference
    # inference_fn_ensemble handles neutral strategy and decoding
    val_preds_quoted = inference_fn_ensemble(val_loader, models, device, val_holdout_df)
    val_preds = [p.strip('"') for p in val_preds_quoted]

    # Compute Metrics
    final_scores = []
    text_lens = []
    sentiments_encoded = []

    for i, row in val_holdout_df.iterrows():
        target = normalize_text(str(row["selected_text"]))
        pred = val_preds[i]
        score = jaccard(pred, target)
        final_scores.append(score)

        # Features
        text_lens.append(len(str(row["text"])))
        sent = row["sentiment"]
        if sent == "negative":
            s_code = 0
        elif sent == "neutral":
            s_code = 1
        else:
            s_code = 2
        sentiments_encoded.append(s_code)

    final_metric = np.mean(final_scores)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")
    errors = 1.0 - np.array(final_scores)
    text_lens = np.array(text_lens)
    sentiments_encoded = np.array(sentiments_encoded)

    if len(errors) > 1:
        corr_len = np.corrcoef(errors, text_lens)[0, 1]
        corr_sent = np.corrcoef(errors, sentiments_encoded)[0, 1]
    else:
        corr_len = 0.0
        corr_sent = 0.0

    print(f"Correlation (Error vs Text Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Sentiment): {corr_sent:.4f}")

    # 5. Submission Generation
    if final_metric > 0.7093:
        print("Validation metric threshold met. Generating submission...")

        test_df = pd.read_csv(Config.TEST_META)
        test_loader = get_eval_loader(test_df, prefix="test")

        predictions = inference_fn_ensemble(test_loader, models, device, test_df)

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
