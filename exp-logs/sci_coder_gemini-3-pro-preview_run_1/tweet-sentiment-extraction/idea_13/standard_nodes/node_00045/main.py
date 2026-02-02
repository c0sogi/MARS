import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, AverageMeter, normalize_text, jaccard
from library.data import get_loaders, get_test_loader
from library.model import TweetModel
from library.engine import train_fn, eval_fn, get_optimizer_params


def predict_loader(model, loader, device):
    """
    Runs inference on a loader and returns a dictionary of {textID: predicted_string}.
    Handles text normalization for correct offset extraction.
    """
    model.eval()
    preds = {}

    # We need to access the original textIDs to map predictions back
    # The loader dataset has texts, but not textIDs directly accessible in the batch unless we modify dataset
    # However, the loader is sequential. We can map by index if we have the dataframe.
    # But TweetDataset doesn't return textID.
    # Workaround: The loader provided by get_loaders corresponds to the validation dataframe for that fold.
    # We will assume sequential order matches.

    all_preds = []

    with torch.no_grad():
        for data in loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            offsets = data["offsets"].cpu().numpy()
            texts = data["text"]  # Raw texts

            start_logits, end_logits = model(input_ids, attention_mask)

            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()
            mask_np = attention_mask.cpu().numpy()

            for i in range(len(input_ids)):
                text = texts[i]
                # Normalize text to align with tokenizer offsets
                norm_text = normalize_text(text)

                start_logit = start_logits[i]
                end_logit = end_logits[i]
                offset = offsets[i]
                m = mask_np[i]

                # Mask padding
                start_logit[m == 0] = -1e9
                end_logit[m == 0] = -1e9

                # Joint Decoding
                sum_matrix = start_logit[:, None] + end_logit[None, :]
                upper_tri_mask = np.triu(np.ones_like(sum_matrix))
                sum_matrix = np.where(upper_tri_mask, sum_matrix, -1e9)

                max_idx = np.argmax(sum_matrix)
                start_idx, end_idx = np.unravel_index(max_idx, sum_matrix.shape)

                if start_idx < len(offset) and end_idx < len(offset):
                    char_start = offset[start_idx][0]
                    char_end = offset[end_idx][1]
                    pred_text = norm_text[char_start:char_end]
                else:
                    pred_text = norm_text

                all_preds.append(pred_text)

    return all_preds


def run_training():
    seed_everything(Config.SEED)

    print(
        f"Starting training with {Config.N_FOLDS} folds, {Config.EPOCHS} epochs each."
    )

    # Store OOF predictions: list of (textID, pred_text)
    # Since we don't have textIDs in the loader, we will align with the validation dataframe
    oof_map = {}

    for fold in range(Config.N_FOLDS):
        print(f"\n=== Fold {fold} ===")
        train_loader, val_loader = get_loaders(fold)

        device = torch.device(Config.DEVICE)
        model = TweetModel()
        model.to(device)

        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=Config.LEARNING_RATE,
            decoder_lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        optimizer = AdamW(optimizer_parameters, lr=Config.LEARNING_RATE, eps=Config.EPS)

        num_train_steps = int(len(train_loader) * Config.EPOCHS)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        best_jaccard = -1
        best_model_path = os.path.join(
            Config.WORKING_DIR, f"best_model_fold_{fold}.bin"
        )

        for epoch in range(Config.EPOCHS):
            train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
            val_loss, val_jaccard = eval_fn(val_loader, model, device)

            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
            )

            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved for fold {fold}")

        # Load best model for OOF inference
        print(f"Loading best model for fold {fold} inference...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Get predictions
        fold_preds = predict_loader(model, val_loader, device)

        # Map back to textIDs
        # We need to reconstruct the validation dataframe for this fold to get IDs
        # get_loaders does splitting internally, so we replicate the logic to get IDs
        # Or simpler: we read the full metadata and split using the same seed
        train_df = pd.read_csv(Config.TRAIN_FILE)
        val_df = pd.read_csv("./metadata/validation_metadata.csv")
        full_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        full_df = full_df.dropna(subset=["text", "sentiment", "selected_text"])

        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )
        splits = list(skf.split(full_df, full_df["sentiment"]))
        _, val_idx = splits[fold]
        val_data = full_df.iloc[val_idx].reset_index(drop=True)

        # Filter neutrals as get_loaders does
        val_data_filtered = val_data[val_data["sentiment"] != "neutral"].reset_index(
            drop=True
        )

        # Store in OOF map
        for idx, row in val_data_filtered.iterrows():
            oof_map[row["textID"]] = fold_preds[idx]

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    return oof_map


def validate_and_analyze(oof_map):
    print("\n=== Final Validation & Failure Analysis ===")
    val_meta_path = "./metadata/validation_metadata.csv"
    val_df = pd.read_csv(val_meta_path)

    final_preds = []
    jaccard_scores = []

    for _, row in val_df.iterrows():
        text_id = row["textID"]
        sentiment = row["sentiment"]
        text = str(row["text"])
        selected_text = str(row["selected_text"])

        if sentiment == "neutral":
            pred = text  # Identity mapping for neutral
        else:
            # Retrieve from OOF, fallback to text if missing (shouldn't happen)
            pred = oof_map.get(text_id, text)

        score = jaccard(selected_text, pred)

        final_preds.append(pred)
        jaccard_scores.append(score)

    val_df["prediction"] = final_preds
    val_df["jaccard"] = jaccard_scores
    val_df["error"] = 1.0 - val_df["jaccard"]
    val_df["text_len"] = val_df["text"].astype(str).apply(len)

    final_metric = np.mean(jaccard_scores)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    correlation = val_df[["error", "text_len"]].corr().iloc[0, 1]
    print(f"Correlation between Error and Text Length: {correlation:.4f}")

    return final_metric


def generate_submission():
    print("\n=== Generating Submission ===")
    test_df = pd.read_csv(Config.TEST_FILE)
    test_loader = get_test_loader(test_df)
    device = torch.device(Config.DEVICE)

    # Initialize aggregated logits
    # We need to know the size. test_loader batches.
    # We will accumulate logits in a list of arrays matching the dataset

    # First pass to get shapes and placeholders
    # Actually, simpler to iterate loader and predict with all models

    # Load all models
    models = []
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"best_model_fold_{fold}.bin")
        model = TweetModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    all_preds = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Inference")):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            offsets = data["offsets"].cpu().numpy()
            texts = data["text"]
            sentiments = data["sentiment"]

            # Ensemble Logits
            avg_start_logits = None
            avg_end_logits = None

            for model in models:
                start_logits, end_logits = model(input_ids, attention_mask)
                if avg_start_logits is None:
                    avg_start_logits = start_logits
                    avg_end_logits = end_logits
                else:
                    avg_start_logits += start_logits
                    avg_end_logits += end_logits

            avg_start_logits /= len(models)
            avg_end_logits /= len(models)

            avg_start_logits = avg_start_logits.cpu().numpy()
            avg_end_logits = avg_end_logits.cpu().numpy()
            mask_np = attention_mask.cpu().numpy()

            for i in range(len(input_ids)):
                text = texts[i]
                sentiment = sentiments[i]

                # Neutral Strategy
                if sentiment == "neutral":
                    all_preds.append(text)
                    continue

                # Non-Neutral Strategy
                norm_text = normalize_text(text)
                start_logit = avg_start_logits[i]
                end_logit = avg_end_logits[i]
                offset = offsets[i]
                m = mask_np[i]

                start_logit[m == 0] = -1e9
                end_logit[m == 0] = -1e9

                sum_matrix = start_logit[:, None] + end_logit[None, :]
                upper_tri_mask = np.triu(np.ones_like(sum_matrix))
                sum_matrix = np.where(upper_tri_mask, sum_matrix, -1e9)

                max_idx = np.argmax(sum_matrix)
                start_idx, end_idx = np.unravel_index(max_idx, sum_matrix.shape)

                if start_idx < len(offset) and end_idx < len(offset):
                    char_start = offset[start_idx][0]
                    char_end = offset[end_idx][1]
                    pred_text = norm_text[char_start:char_end]
                else:
                    pred_text = norm_text

                all_preds.append(pred_text)

    # Clean up models
    for m in models:
        del m
    torch.cuda.empty_cache()

    # Create submission
    sub_df = pd.DataFrame({"textID": test_df["textID"], "selected_text": all_preds})

    # Ensure quotes are handled if necessary by to_csv, but format requirements say:
    # textID,selected_text
    # 2,"very good"
    # standard csv quoting should handle this
    sub_df.to_csv(
        Config.SUBMISSION_FILE, index=False, quoting=1
    )  # quoting=1 is QUOTE_ALL/QUOTE_NONNUMERIC often preferred
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    # 1. Train and get OOF
    oof_predictions = run_training()

    # 2. Validate
    metric = validate_and_analyze(oof_predictions)

    # 3. Submit if good enough
    if metric > 0.7093:
        generate_submission()
    else:
        print("Validation metric too low. Skipping submission.")
