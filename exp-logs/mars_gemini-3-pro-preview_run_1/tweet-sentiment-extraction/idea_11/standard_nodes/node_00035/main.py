import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_cosine_schedule_with_warmup

# Import library modules
from library.config import Config
from library.dataset import get_data, TweetDataset
from library.model import SentimentModel
from library.engine import train_fn, eval_fn
from library.utils import seed_everything, jaccard, get_best_start_end_idxs
from library.inference import run_inference

# Suppress warnings
warnings.filterwarnings("ignore")


def run_validation_ensemble(val_df, model_dir, n_folds, device):
    """
    Runs ensemble inference on the validation set and computes the Jaccard score.
    """
    print("\nRunning Ensemble Validation on Hold-out Set...")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Prepare Validation Dataset
    # We don't filter neutrals here because we need to evaluate on the full set
    # We pass split_name="val_eval" to avoid cache conflicts
    val_dataset = get_data(
        val_df, tokenizer, split_name="val_eval", load_cached_data=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    num_samples = len(val_dataset)
    max_len = Config.MAX_LEN

    avg_start_logits = np.zeros((num_samples, max_len), dtype=np.float32)
    avg_end_logits = np.zeros((num_samples, max_len), dtype=np.float32)

    models_used = 0

    for fold in range(n_folds):
        model_path = os.path.join(model_dir, f"model_fold_{fold}.bin")
        if not os.path.exists(model_path):
            continue

        model = SentimentModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_start_preds = []
        fold_end_preds = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, dtype=torch.long)
                attention_mask = batch["attention_mask"].to(device, dtype=torch.long)

                with torch.amp.autocast("cuda", enabled=Config.USE_AMP):
                    start_logits, end_logits = model(input_ids, attention_mask)

                fold_start_preds.append(start_logits.float().cpu().numpy())
                fold_end_preds.append(end_logits.float().cpu().numpy())

        avg_start_logits += np.concatenate(fold_start_preds, axis=0)
        avg_end_logits += np.concatenate(fold_end_preds, axis=0)
        models_used += 1

        del model
        torch.cuda.empty_cache()

    if models_used == 0:
        return 0.0, pd.DataFrame()

    avg_start_logits /= models_used
    avg_end_logits /= models_used

    # Decode
    jaccard_scores = []

    texts = val_dataset.texts
    selected_texts = val_dataset.selected_texts
    sentiments = val_dataset.sentiments
    offsets = val_dataset.offsets
    ids = val_dataset.ids

    results = []

    for i in range(num_samples):
        text = texts[i]
        selected_text = selected_texts[i]
        sentiment = sentiments[i]
        offset = offsets[i]

        if sentiment == "neutral":
            pred_text = text
        else:
            idx_start, idx_end = get_best_start_end_idxs(
                avg_start_logits[i], avg_end_logits[i]
            )
            char_start = offset[idx_start][0]
            char_end = offset[idx_end][1]
            pred_text = text[char_start:char_end]

        score = jaccard(selected_text, pred_text)
        jaccard_scores.append(score)

        results.append(
            {
                "textID": ids[i],
                "text": text,
                "selected_text": selected_text,
                "prediction": pred_text,
                "sentiment": sentiment,
                "jaccard": score,
                "text_len": len(text),
            }
        )

    final_score = np.mean(jaccard_scores)
    return final_score, pd.DataFrame(results)


def main():
    # 1. Configuration and Setup
    # Override Config for Fast Baseline
    Config.EPOCHS = 2
    Config.N_FOLDS = 3  # Reduced from 5 to 3 for speed
    Config.TRAIN_BATCH_SIZE = 16  # Increase batch size slightly for A100 to speed up

    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Starting Run with Config: {Config.NAME}")
    print(f"Device: {device}")
    print(f"Epochs: {Config.EPOCHS}, Folds: {Config.N_FOLDS}")

    # 2. Data Loading
    print("Loading Metadata...")
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError("Train metadata not found.")
    if not os.path.exists(Config.VAL_META_PATH):
        raise FileNotFoundError("Validation metadata not found.")

    df_train_full = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_holdout = pd.read_csv(Config.VAL_META_PATH)

    # 3. Cross-Validation Training
    # We split the full training set into folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    for fold, (train_idx, _) in enumerate(
        skf.split(df_train_full, df_train_full["sentiment"])
    ):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Prepare Train Data for this fold
        df_train_fold = df_train_full.iloc[train_idx].reset_index(drop=True)

        # Filter Neutrals explicitly here to ensure control
        if Config.TRAIN_EXCLUDE_NEUTRAL:
            df_train_fold = df_train_fold[
                df_train_fold["sentiment"] != "neutral"
            ].reset_index(drop=True)

        train_dataset = get_data(
            df_train_fold,
            tokenizer,
            split_name=f"train_fold_{fold}",
            load_cached_data=False,  # Disable cache to ensure correct fold data
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # Initialize Model
        model = SentimentModel()
        model.to(device)

        # Optimizer and Scheduler
        optimizer_parameters = model.get_optimizer_params(
            encoder_lr=Config.LEARNING_RATE,
            decoder_lr=Config.LEARNING_RATE,  # Usually head LR is same or higher
            weight_decay=Config.WEIGHT_DECAY,
        )
        optimizer = AdamW(optimizer_parameters, lr=Config.LEARNING_RATE, eps=1e-6)

        num_train_steps = int(
            len(train_dataset) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        # Training Loop
        for epoch in range(Config.EPOCHS):
            avg_loss = train_fn(train_loader, model, optimizer, device, scheduler)
            print(f"Fold {fold} | Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

        # Save Model
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.bin")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        del model, optimizer, scheduler, train_loader, train_dataset
        torch.cuda.empty_cache()

    # 4. Validation Assessment
    val_score, val_results = run_validation_ensemble(
        df_val_holdout, Config.WORKING_DIR, Config.N_FOLDS, device
    )

    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    if not val_results.empty:
        val_results["error"] = 1.0 - val_results["jaccard"]

        # Correlation with Text Length
        corr_len = val_results["text_len"].corr(val_results["error"])
        print(f"Correlation (Error vs Text Length): {corr_len:.4f}")

        # Error by Sentiment
        print("Mean Jaccard by Sentiment:")
        print(val_results.groupby("sentiment")["jaccard"].mean())

        # Top Errors
        print("\nTop 3 Worst Predictions:")
        worst = val_results.sort_values("jaccard").head(3)
        for _, row in worst.iterrows():
            print(f"Text: {row['text']}")
            print(f"True: {row['selected_text']}")
            print(f"Pred: {row['prediction']}")
            print(f"Score: {row['jaccard']:.4f}\n")

    # 6. Submission
    THRESHOLD = 0.7043342108129372
    if val_score > THRESHOLD:
        print(
            f"\nValidation score ({val_score:.5f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        run_inference(
            test_meta_path=Config.TEST_META_PATH,
            base_model_dir=Config.WORKING_DIR,
            output_path=Config.SUBMISSION_PATH,
            n_folds=Config.N_FOLDS,
            device=device,
        )
    else:
        print(
            f"\nValidation score ({val_score:.5f}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
