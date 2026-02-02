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
    Config.DEBUG = False

    # Use Config.EPOCHS (3) and Config.N_FOLDS (3)
    print(f"Starting execution on device: {device}")

    full_train_df = pd.read_csv(Config.TRAIN_META)
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(full_train_df, full_train_df["sentiment"]))

    # Accumulate OOF results
    final_scores = []
    text_lens = []
    sentiments_encoded = []

    # Store model paths for ensemble
    model_paths = []

    # Loop through folds
    for fold in range(Config.N_FOLDS):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Define model path for this fold
        fold_model_path = os.path.join(
            Config.WORKING_DIR, f"best_model_fold_{fold}.bin"
        )
        model_paths.append(fold_model_path)

        # Get loaders
        train_loader, val_loader = get_data_loaders(fold=fold, load_cached_data=True)

        # Get validation dataframe for this fold (Active Only for Training Eval)
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

        best_jaccard = 0.0

        # Training Loop
        for epoch in range(Config.EPOCHS):
            train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
            val_loss, val_jaccard = eval_fn(val_loader, model, device, val_df_active)

            # print(f"Fold {fold+1} Epoch {epoch+1} | Val Jaccard: {val_jaccard:.4f}")

            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                torch.save(model.state_dict(), fold_model_path)

        # --- OOF Evaluation for this Fold ---
        print(f"Performing OOF Validation for Fold {fold+1}...")
        model.load_state_dict(torch.load(fold_model_path, map_location=device))
        model.eval()

        # Get FULL validation set for this fold
        val_df_full = full_train_df.iloc[val_idx].reset_index(drop=True)
        val_df_active_full = val_df_full[
            val_df_full["sentiment"] != "neutral"
        ].reset_index(drop=True)
        val_df_neutral_full = val_df_full[
            val_df_full["sentiment"] == "neutral"
        ].reset_index(drop=True)

        # 1. Active Predictions
        active_preds_quoted = inference_fn(
            val_loader, model, device, val_df_active_full
        )
        active_preds = [p.strip('"') for p in active_preds_quoted]

        for i, pred in enumerate(active_preds):
            target = normalize_text(str(val_df_active_full.iloc[i]["selected_text"]))
            score = jaccard(pred, target)
            final_scores.append(score)

            txt = str(val_df_active_full.iloc[i]["text"])
            text_lens.append(len(txt))
            sent = val_df_active_full.iloc[i]["sentiment"]
            sentiments_encoded.append(0 if sent == "negative" else 2)

        # 2. Neutral Predictions (Identity)
        for i, row in val_df_neutral_full.iterrows():
            text = normalize_text(str(row["text"]))
            target = normalize_text(str(row["selected_text"]))
            score = jaccard(text, target)
            final_scores.append(score)

            text_lens.append(len(str(row["text"])))
            sentiments_encoded.append(1)

        # Clean up
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 5. Final Validation Assessment (Aggregated OOF)
    final_metric = np.mean(final_scores)
    print(f"Final Validation Metric (OOF): {final_metric:.16f}")

    # 6. Failure Analysis
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

    # 7. Submission Generation (Ensemble)
    if final_metric > 0.7093:
        print("Validation metric threshold met. Generating submission with Ensemble...")
        test_loader, test_df = get_test_loader(load_cached_data=True)

        # We need to manually ensemble logits
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

        # Collect logits from all models
        avg_start_logits = None
        avg_end_logits = None
        offsets_list = []

        # Get offsets once (same for all models)
        with torch.no_grad():
            for d in test_loader:
                offsets_list.append(d["offsets"].cpu().numpy())
        offsets_preds = np.concatenate(offsets_list)

        for mp in model_paths:
            print(f"Loading {mp} for inference...")
            model = TweetModel()
            model.load_state_dict(torch.load(mp, map_location=device))
            model.to(device)
            model.eval()

            fold_start_logits = []
            fold_end_logits = []

            with torch.no_grad():
                for d in test_loader:
                    input_ids = d["input_ids"].to(device)
                    attention_mask = d["attention_mask"].to(device)
                    s, e = model(input_ids, attention_mask)
                    fold_start_logits.append(s.cpu().numpy())
                    fold_end_logits.append(e.cpu().numpy())

            fold_start = np.concatenate(fold_start_logits)
            fold_end = np.concatenate(fold_end_logits)

            if avg_start_logits is None:
                avg_start_logits = fold_start
                avg_end_logits = fold_end
            else:
                avg_start_logits += fold_start
                avg_end_logits += fold_end

            del model
            torch.cuda.empty_cache()

        # Average
        avg_start_logits /= len(model_paths)
        avg_end_logits /= len(model_paths)

        # Decode
        final_predictions = []
        for i, row in test_df.iterrows():
            text = normalize_text(str(row["text"]))
            sentiment = str(row["sentiment"])

            if sentiment == "neutral":
                final_predictions.append(f'"{text}"')
                continue

            s_logits = avg_start_logits[i]
            e_logits = avg_end_logits[i]
            offsets = offsets_preds[i]

            sum_logits = np.add.outer(s_logits, e_logits)
            mask = np.triu(np.ones_like(sum_logits))
            sum_logits[mask == 0] = -float("inf")
            best_idx = np.argmax(sum_logits)
            best_start, best_end = np.unravel_index(best_idx, sum_logits.shape)

            if best_start < len(offsets) and best_end < len(offsets):
                start_char = offsets[best_start][0]
                end_char = offsets[best_end][1]
                pred_text = text[start_char:end_char]
            else:
                pred_text = text

            final_predictions.append(f'"{pred_text}"')

        submission_df = pd.DataFrame(
            {"textID": test_df["textID"], "selected_text": final_predictions}
        )
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_metric:.4f} did not meet threshold 0.7093. Submission skipped."
        )


if __name__ == "__main__":
    run()
