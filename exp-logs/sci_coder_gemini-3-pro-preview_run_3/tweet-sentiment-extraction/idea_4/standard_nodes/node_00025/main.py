import os
import sys
import gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, AverageMeter, jaccard, get_selected_text
from library.dataset import get_data, TweetDataset
from library.model import TweetModel
from library.engine import train_fn, eval_fn


def run_training():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    # Load metadata
    if not os.path.exists(Config.TRAIN_META_PATH) or not os.path.exists(
        Config.VAL_META_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure metadata generation was successful."
        )

    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)

    # Combine for full CV
    df_full = pd.concat([df_train_meta, df_val_meta]).reset_index(drop=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # CV Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for OOF analysis
    oof_preds = []
    oof_truths = []
    oof_sentiments = []
    oof_texts = []
    oof_jaccards = []

    # 3. Training Loop
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_full, df_full["sentiment"])
    ):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Split Data
        df_train = df_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_full.iloc[val_idx].reset_index(drop=True)

        # Prepare Datasets
        # Train: is_train=True, filter_neutral=True (as per config/idea)
        train_ds = get_data(
            df_train,
            tokenizer,
            Config.MAX_LEN,
            os.path.join(Config.WORKING_DIR, f"fold_{fold}_train"),
            load_cached_data=True,
            is_train=True,
            filter_neutral=Config.FILTER_NEUTRAL,
        )

        # Val: is_train=True (need labels for eval), filter_neutral=False (evaluate on all)
        val_ds = get_data(
            df_val,
            tokenizer,
            Config.MAX_LEN,
            os.path.join(Config.WORKING_DIR, f"fold_{fold}_val"),
            load_cached_data=True,
            is_train=True,
            filter_neutral=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model Setup
        model = TweetModel()
        model.to(device)

        # Optimizer & Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        num_train_steps = int(len(train_ds) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        # Initialize Scaler for AMP
        scaler = GradScaler(enabled=torch.cuda.is_available())

        # Training
        best_jaccard = 0
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.bin")

        for epoch in range(Config.EPOCHS):
            train_loss = train_fn(
                train_loader, model, optimizer, device, scheduler, scaler=scaler
            )
            val_loss, val_jaccard = eval_fn(val_loader, model, device)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
            )

            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                torch.save(model.state_dict(), model_path)
                print(f"  -> Saved best model for fold {fold+1}")

        # Load best model for OOF generation
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Generate OOF predictions for this fold manually to store details
        with torch.no_grad():
            for d in val_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                token_type_ids = d.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                start_logits, end_logits = model(
                    input_ids, attention_mask, token_type_ids
                )

                start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
                end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

                orig_texts = d["orig_text"]
                sentiments = d["sentiment"]
                offsets = d["offsets"].numpy()
                selected_texts = d["selected_text"]

                for i in range(len(orig_texts)):
                    text = orig_texts[i]
                    sp = start_probs[i]
                    ep = end_probs[i]
                    sentiment = sentiments[i]
                    offset = offsets[i]
                    gt = selected_texts[i]

                    pred = get_selected_text(text, sp, ep, sentiment, offset)
                    score = jaccard(pred, gt)

                    oof_preds.append(pred)
                    oof_truths.append(gt)
                    oof_sentiments.append(sentiment)
                    oof_texts.append(text)
                    oof_jaccards.append(score)

        # Cleanup to free memory
        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Validation Assessment
    overall_jaccard = np.mean(oof_jaccards)
    print(f"\nFinal Validation Metric: {overall_jaccard}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_oof = pd.DataFrame(
        {
            "text": oof_texts,
            "sentiment": oof_sentiments,
            "selected_text": oof_truths,
            "prediction": oof_preds,
            "jaccard": oof_jaccards,
        }
    )

    df_oof["text_len"] = df_oof["text"].apply(len)
    df_oof["error"] = 1.0 - df_oof["jaccard"]

    # Correlation
    corr, _ = pearsonr(df_oof["text_len"], df_oof["error"])
    print(f"Correlation between Text Length and Error (1-Jaccard): {corr:.4f}")

    # By Sentiment
    print("Jaccard by Sentiment:")
    print(df_oof.groupby("sentiment")["jaccard"].mean())

    return overall_jaccard


def predict_test():
    print("\n--- Generating Submission ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    df_test = pd.read_csv(Config.TEST_META_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Note: is_train=False means no labels, filter_neutral=False means keep all rows
    test_ds = get_data(
        df_test,
        tokenizer,
        Config.MAX_LEN,
        os.path.join(Config.WORKING_DIR, "test_data"),
        load_cached_data=True,
        is_train=False,
        filter_neutral=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load all models for ensemble
    models = []
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.bin")
        model = TweetModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    final_preds = []

    # Inference
    with torch.no_grad():
        for d in test_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Ensemble: Average Logits
            avg_start_logits = None
            avg_end_logits = None

            for model in models:
                start_logits, end_logits = model(
                    input_ids, attention_mask, token_type_ids
                )

                if avg_start_logits is None:
                    avg_start_logits = start_logits
                    avg_end_logits = end_logits
                else:
                    avg_start_logits += start_logits
                    avg_end_logits += end_logits

            avg_start_logits /= len(models)
            avg_end_logits /= len(models)

            # Decode
            start_probs = torch.softmax(avg_start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(avg_end_logits, dim=1).cpu().numpy()

            orig_texts = d["orig_text"]
            sentiments = d["sentiment"]
            offsets = d["offsets"].numpy()

            for i in range(len(orig_texts)):
                text = orig_texts[i]
                sp = start_probs[i]
                ep = end_probs[i]
                sentiment = sentiments[i]
                offset = offsets[i]

                # Apply decoding logic (Neutral Rule + Summation Decoding)
                pred = get_selected_text(text, sp, ep, sentiment, offset)
                final_preds.append(pred)

    # Create Submission
    # Ensure alignment with original test IDs
    # Since DataLoader preserves order and we didn't shuffle, this aligns with df_test
    submission = pd.DataFrame(
        {"textID": df_test["textID"], "selected_text": final_preds}
    )

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    val_score = run_training()

    # Threshold check
    THRESHOLD = 0.7164761348654044
    if val_score > THRESHOLD:
        predict_test()
    else:
        print(
            f"Validation score {val_score} did not meet threshold {THRESHOLD}. Skipping submission."
        )
