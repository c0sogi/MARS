import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

# Import provided library modules
from library.config import Config
from library.utils import set_seed, jaccard, decode_span
from library.dataset import TweetDataset, get_tokenizer, prepare_data, prepare_test_data
from library.model import TweetModel
from library.engine import train_fn, eval_fn


def run():
    # 1. Initialization
    config = Config()
    set_seed(config.seed)

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Device: {config.device}")

    # 2. Data Preparation
    # Returns dataframe with 'fold' column, filtered for training (no neutrals, no impossible targets)
    df_folds = prepare_data(config, load_cached_data=True)
    tokenizer = get_tokenizer(config)

    # Store OOF predictions: list of dicts {textID, selected_text_pred}
    oof_preds = []

    # 3. Training Loop
    for fold in range(config.n_folds):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Data
        train_df = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
        valid_df = df_folds[df_folds["fold"] == fold].reset_index(drop=True)

        # Datasets
        train_dataset = TweetDataset(train_df, tokenizer, config.max_len, is_test=False)
        valid_dataset = TweetDataset(valid_df, tokenizer, config.max_len, is_test=False)

        # Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Model
        model = TweetModel(config)
        model.to(config.device)

        # Optimization
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        num_train_steps = int(len(train_df) / config.train_batch_size * config.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Loss (CrossEntropy with Label Smoothing)
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # Training State
        best_jaccard = -1.0
        model_save_path = os.path.join(config.output_dir, f"model_fold_{fold}.bin")

        # Epoch Loop
        for epoch in range(config.epochs):
            model.train()
            losses = []
            scaler = GradScaler()

            # Efficient Training Loop with AMP
            for batch in train_loader:
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                token_type_ids = batch["token_type_ids"].to(config.device)
                start_positions = batch["start_positions"].to(config.device)
                end_positions = batch["end_positions"].to(config.device)

                optimizer.zero_grad()

                with autocast():
                    start_logits, end_logits = model(
                        input_ids, attention_mask, token_type_ids
                    )
                    loss = criterion(start_logits, start_positions) + criterion(
                        end_logits, end_positions
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()

                if scheduler:
                    scheduler.step()

                losses.append(loss.item())

            avg_train_loss = np.mean(losses)

            # Validation (using engine's eval_fn)
            avg_val_loss, avg_val_jaccard = eval_fn(
                valid_loader, model, config.device, criterion
            )

            print(
                f"Fold {fold} | Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val Jaccard: {avg_val_jaccard:.5f}"
            )

            if avg_val_jaccard > best_jaccard:
                best_jaccard = avg_val_jaccard
                torch.save(model.state_dict(), model_save_path)

        # Inference on Validation Fold (OOF) for non-neutral samples
        # Load best model
        model.load_state_dict(torch.load(model_save_path, map_location=config.device))
        model.eval()

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                token_type_ids = batch["token_type_ids"].to(config.device)
                ids = batch["textID"]
                texts = batch["text"]
                offsets = batch["offsets"].numpy()

                start_logits, end_logits = model(
                    input_ids, attention_mask, token_type_ids
                )

                start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
                end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

                for i in range(len(ids)):
                    idx_start, idx_end = decode_span(start_probs[i], end_probs[i])

                    if idx_start < len(offsets[i]) and idx_end < len(offsets[i]):
                        char_start = offsets[i][idx_start][0]
                        char_end = offsets[i][idx_end][1]
                        pred = texts[i][char_start:char_end]
                    else:
                        pred = ""

                    oof_preds.append({"textID": ids[i], "selected_text_pred": pred})

        # Cleanup to save memory
        del model, optimizer, scheduler, scaler, train_loader, valid_loader
        torch.cuda.empty_cache()

    # 4. Global Validation & Failure Analysis
    print("\n=== Final Validation & Failure Analysis ===")

    # Load official validation set (contains neutrals)
    val_df_official = pd.read_csv(config.val_path)

    # Map OOF predictions (contains only non-neutrals that passed filtering)
    pred_map = {item["textID"]: item["selected_text_pred"] for item in oof_preds}

    scores = []
    errors = []
    text_lengths = []

    for _, row in val_df_official.iterrows():
        tid = row["textID"]
        sentiment = row["sentiment"]
        text = str(row["text"])
        selected_gt = str(row["selected_text"])

        # Logic: Neutral -> text, Else -> Model Prediction
        if sentiment == "neutral":
            pred = text
        else:
            # If ID not in pred_map (filtered out during training due to impossible alignment), fallback to text
            pred = pred_map.get(tid, text)

        score = jaccard(selected_gt, pred)
        scores.append(score)

        errors.append(1.0 - score)
        text_lengths.append(len(text))

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric:.16f}")

    correlation = np.corrcoef(errors, text_lengths)[0, 1]
    print(f"Correlation between Error and Text Length: {correlation:.6f}")

    # 5. Submission
    THRESHOLD = 0.7093372235447927

    if final_metric > THRESHOLD:
        print("\n=== Generating Submission ===")

        test_df = prepare_test_data(config)
        test_dataset = TweetDataset(test_df, tokenizer, config.max_len, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Ensemble Inference
        # Accumulate logits: (N_samples, Seq_Len)
        avg_start_logits = None
        avg_end_logits = None

        for fold in range(config.n_folds):
            print(f"Inference Fold {fold}...")
            model = TweetModel(config)
            model.load_state_dict(
                torch.load(
                    os.path.join(config.output_dir, f"model_fold_{fold}.bin"),
                    map_location=config.device,
                )
            )
            model.to(config.device)
            model.eval()

            fold_s_logits = []
            fold_e_logits = []

            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(config.device)
                    attention_mask = batch["attention_mask"].to(config.device)
                    token_type_ids = batch["token_type_ids"].to(config.device)

                    s, e = model(input_ids, attention_mask, token_type_ids)
                    fold_s_logits.append(s.cpu().numpy())
                    fold_e_logits.append(e.cpu().numpy())

            fold_s_logits = np.concatenate(fold_s_logits, axis=0)
            fold_e_logits = np.concatenate(fold_e_logits, axis=0)

            if avg_start_logits is None:
                avg_start_logits = fold_s_logits
                avg_end_logits = fold_e_logits
            else:
                avg_start_logits += fold_s_logits
                avg_end_logits += fold_e_logits

            del model
            torch.cuda.empty_cache()

        # Average logits
        avg_start_logits /= config.n_folds
        avg_end_logits /= config.n_folds

        # Decode
        final_preds = []

        # Iterate loader again to get metadata for decoding
        current_idx = 0
        for batch in test_loader:
            texts = batch["text"]
            sentiments = batch["sentiment"]
            offsets = batch["offsets"].numpy()
            batch_len = len(texts)

            for i in range(batch_len):
                global_idx = current_idx + i

                if sentiments[i] == "neutral":
                    pred = texts[i]
                else:
                    s_probs = torch.softmax(
                        torch.tensor(avg_start_logits[global_idx]), dim=0
                    ).numpy()
                    e_probs = torch.softmax(
                        torch.tensor(avg_end_logits[global_idx]), dim=0
                    ).numpy()

                    idx_start, idx_end = decode_span(s_probs, e_probs)

                    if idx_start < len(offsets[i]) and idx_end < len(offsets[i]):
                        char_start = offsets[i][idx_start][0]
                        char_end = offsets[i][idx_end][1]
                        pred = texts[i][char_start:char_end]
                    else:
                        pred = texts[i]

                final_preds.append(pred)

            current_idx += batch_len

        # Save Submission
        sub_df = pd.DataFrame(
            {"textID": test_df["textID"], "selected_text": final_preds}
        )

        # Using QUOTE_NONNUMERIC ensures strings are quoted, satisfying the format requirement
        sub_df.to_csv(config.submission_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Submission saved to {config.submission_path}")


if __name__ == "__main__":
    run()
