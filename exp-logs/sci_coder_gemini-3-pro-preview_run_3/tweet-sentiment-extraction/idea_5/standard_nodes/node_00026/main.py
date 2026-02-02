import os
import sys
import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import process_data, TweetDataset
from library.model import TweetModel
from library.engine import train_fn


def get_predictions(models, data_loader, df, tokenizer, device):
    """
    Generates predictions using an ensemble of models with Hybrid Decoding.
    """
    # Set models to eval mode
    for model in models:
        model.eval()
        model.to(device)

    final_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Ensemble variables
            avg_start_logits = None
            avg_end_logits = None
            avg_content_logits = None

            # Accumulate logits from all models
            for model in models:
                s, e, c = model(input_ids, attention_mask)

                if avg_start_logits is None:
                    avg_start_logits = s
                    avg_end_logits = e
                    avg_content_logits = c
                else:
                    avg_start_logits += s
                    avg_end_logits += e
                    avg_content_logits += c

            # Average logits
            n_models = len(models)
            avg_start_logits /= n_models
            avg_end_logits /= n_models
            avg_content_logits /= n_models

            # Convert to probabilities
            p_start = torch.softmax(avg_start_logits, dim=1).cpu().numpy()
            p_end = torch.softmax(avg_end_logits, dim=1).cpu().numpy()
            p_content = torch.sigmoid(avg_content_logits).cpu().numpy()

            ids = input_ids.cpu().numpy()

            # Map back to dataframe rows to check sentiment
            start_row_idx = batch_idx * data_loader.batch_size

            for i in range(len(ids)):
                global_idx = start_row_idx + i
                row = df.iloc[global_idx]
                sentiment = row["sentiment"]
                original_text = str(row["text"])

                # Deterministic Rule: Neutral -> Full Text
                if sentiment == "neutral":
                    final_predictions.append(original_text)
                    continue

                # Hybrid Decoding Strategy
                ps = p_start[i]
                pe = p_end[i]
                pc = p_content[i]

                best_score = -float("inf")
                best_start = 0
                best_end = 0
                seq_len = len(ps)

                # Search for span maximizing: P_start + P_end + Mean_Content
                # Optimization: Skip low probability start/end tokens
                for s_idx in range(seq_len):
                    if ps[s_idx] < 0.001:
                        continue

                    for e_idx in range(s_idx, seq_len):
                        if pe[e_idx] < 0.001:
                            continue

                        content_score = np.mean(pc[s_idx : e_idx + 1])
                        score = ps[s_idx] + pe[e_idx] + content_score

                        if score > best_score:
                            best_score = score
                            best_start = s_idx
                            best_end = e_idx

                # Fallback if no valid span found (rare)
                if best_score == -float("inf"):
                    best_start = np.argmax(ps)
                    best_end = np.argmax(pe)
                    if best_end < best_start:
                        best_end = best_start

                pred_ids = ids[i][best_start : best_end + 1]
                pred_str = tokenizer.decode(pred_ids, skip_special_tokens=True)
                final_predictions.append(pred_str)

    return final_predictions


def run():
    # 1. Setup
    # Override epochs to 2 for a faster baseline execution while maintaining performance
    Config.setup(epochs=2)
    seed_everything(Config.SEED)
    device = Config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # 2. Load Data
    print("Loading Metadata...")
    train_full = pd.read_csv(Config.TRAIN_FILE)
    val_holdout = pd.read_csv(Config.VAL_FILE)
    test_df = pd.read_csv(Config.TEST_FILE)

    # 3. 5-Fold Stratified Cross-Validation Training
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    train_full = train_full.reset_index(drop=True)

    model_paths = []

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_full, train_full["sentiment"])
    ):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Create Fold Splits
        df_train_fold = train_full.iloc[train_idx].reset_index(drop=True)
        df_val_fold = train_full.iloc[val_idx].reset_index(drop=True)

        # Process Data (using unique cache prefixes per fold)
        # Note: process_data implicitly filters out 'neutral' when is_test=False
        t_ids, t_att, t_start, t_end, t_content = process_data(
            df_train_fold,
            tokenizer,
            Config.MAX_LEN,
            is_test=False,
            cache_prefix=f"train_fold_{fold}",
            load_cached_data=True,
        )

        # We process val fold just to have it if needed, though we don't strictly evaluate per-epoch here to save time
        v_ids, v_att, v_start, v_end, v_content = process_data(
            df_val_fold,
            tokenizer,
            Config.MAX_LEN,
            is_test=False,
            cache_prefix=f"val_fold_{fold}",
            load_cached_data=True,
        )

        # Create Datasets and Loaders
        train_ds = TweetDataset(t_ids, t_att, t_start, t_end, t_content)
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = TweetModel()
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Training Loop
        for epoch in range(Config.EPOCHS):
            avg_loss = train_fn(train_loader, model, optimizer, device)
            print(f"  Epoch {epoch+1}/{Config.EPOCHS} - Loss: {avg_loss:.4f}")

        # Save Model
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.bin")
        torch.save(model.state_dict(), model_path)
        model_paths.append(model_path)

        # Cleanup
        del model, optimizer, train_loader, train_ds, t_ids, v_ids
        torch.cuda.empty_cache()

    # 4. Final Validation on Hold-out Set
    print("\n--- Final Validation on Hold-out Set ---")

    # Load all models for ensemble
    models = []
    for path in model_paths:
        m = TweetModel()
        m.load_state_dict(torch.load(path, map_location=device))
        models.append(m)

    # Prepare Validation Data
    # Use is_test=True to keep neutral rows and get inference-ready format
    val_ids, val_att, _, _, _ = process_data(
        val_holdout,
        tokenizer,
        Config.MAX_LEN,
        is_test=True,
        cache_prefix="val_holdout_final",
        load_cached_data=True,
    )

    val_ds = TweetDataset(val_ids, val_att)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Generate Predictions
    val_preds = get_predictions(models, val_loader, val_holdout, tokenizer, device)

    # Compute Metric
    val_holdout["prediction"] = val_preds
    val_holdout["jaccard"] = val_holdout.apply(
        lambda x: jaccard(x["selected_text"], x["prediction"]), axis=1
    )
    final_score = val_holdout["jaccard"].mean()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    val_holdout["error"] = 1.0 - val_holdout["jaccard"]
    val_holdout["text_len_char"] = val_holdout["text"].apply(len)
    val_holdout["text_len_word"] = val_holdout["text"].apply(
        lambda x: len(str(x).split())
    )

    corr_char, _ = pearsonr(val_holdout["error"], val_holdout["text_len_char"])
    corr_word, _ = pearsonr(val_holdout["error"], val_holdout["text_len_word"])

    print(f"Correlation (Error vs Char Length): {corr_char:.4f}")
    print(f"Correlation (Error vs Word Length): {corr_word:.4f}")

    # 6. Submission
    THRESHOLD = 0.7164761348654044
    if final_score > THRESHOLD:
        print("\n--- Generating Submission ---")

        # Process Test Data
        test_ids, test_att, _, _, _ = process_data(
            test_df,
            tokenizer,
            Config.MAX_LEN,
            is_test=True,
            cache_prefix="test_final",
            load_cached_data=True,
        )

        test_ds = TweetDataset(test_ids, test_att)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Predict
        test_preds = get_predictions(models, test_loader, test_df, tokenizer, device)

        submission_df = pd.DataFrame(
            {"textID": test_df["textID"], "selected_text": test_preds}
        )

        # Save with quoting to ensure format compliance
        submission_df.to_csv(
            Config.SUBMISSION_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC
        )
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"Validation score {final_score} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
