import os
import gc
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, logging

# Import from library
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import get_loaders, load_and_cache_data, TweetDataset, get_test_loader
from library.model import TweetModel, get_optimizer_params
from library.engine import train_fn, eval_fn

# Suppress warnings
logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")


def run():
    # Create submission directory
    os.makedirs("./submission", exist_ok=True)

    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Override Config for Fast Baseline
    # 2 Epochs is a good balance for DeBERTa Large to converge sufficiently while remaining fast
    Config.EPOCHS = 2
    Config.TRAIN_BATCH_SIZE = 16  # A100 40GB can handle 16 for Large
    Config.VALID_BATCH_SIZE = 64  # Faster inference

    # 2. Training Loop (5 Folds)
    model_paths = []

    for fold in range(Config.N_FOLDS):
        print(f"\n=== Fold {fold} ===")
        # Load Data
        train_loader, val_loader = get_loaders(fold, load_cached_data=True)

        # Initialize Model
        model = TweetModel()
        model.to(device)

        # Optimizer & Scheduler
        optimizer_grouped_parameters = get_optimizer_params(
            model, Config.LEARNING_RATE, Config.WEIGHT_DECAY, Config.LLRD_DECAY
        )
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=Config.LEARNING_RATE,
            eps=Config.OPTIMIZER_EPS,
        )

        num_train_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        # Train
        best_loss = float("inf")
        best_model_path = os.path.join(Config.OUTPUT_DIR, f"best_model_fold_{fold}.bin")

        for epoch in range(Config.EPOCHS):
            avg_loss = train_fn(train_loader, model, optimizer, device, scheduler)
            val_loss, _, _ = eval_fn(val_loader, model, device)
            print(
                f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

        model_paths.append(best_model_path)

        # Cleanup to free memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Validation on Hold-Out Set
    print("\n=== Validation on Hold-Out Set ===")
    val_data = load_and_cache_data(Config.VAL_FILE, "val", load_cached_data=True)

    val_dataset = TweetDataset(
        input_ids=val_data["input_ids"],
        attention_mask=val_data["attention_mask"],
        offsets=val_data["offsets"],
        texts=val_data["texts"],
        sentiments=val_data["sentiments"],
        start_targets=None,
        end_targets=None,
        selected_texts=val_data.get("selected_texts"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Inference (Ensemble)
    avg_start_logits = np.zeros((len(val_dataset), Config.MAX_LEN), dtype=np.float32)
    avg_end_logits = np.zeros((len(val_dataset), Config.MAX_LEN), dtype=np.float32)

    for fold, path in enumerate(model_paths):
        # print(f"Inference with model fold {fold}...")
        model = TweetModel()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        _, start_logits, end_logits = eval_fn(val_loader, model, device)
        avg_start_logits += start_logits / Config.N_FOLDS
        avg_end_logits += end_logits / Config.N_FOLDS

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Decode Predictions
    predictions = []
    texts = val_data["texts"]
    sentiments = val_data["sentiments"]
    offsets = val_data["offsets"]

    for i in range(len(texts)):
        text = texts[i]
        sentiment = sentiments[i]

        if sentiment == "neutral":
            predictions.append(text)
        else:
            s_log = avg_start_logits[i]
            e_log = avg_end_logits[i]
            off = offsets[i]

            # Mask invalid tokens (where end <= start, e.g. [CLS], [SEP], PAD)
            # off shape: (max_len, 2)
            valid_mask = off[:, 1] > off[:, 0]

            s_log[~valid_mask] = -1e9
            e_log[~valid_mask] = -1e9

            # Joint Logit Decoding
            score_mat = s_log[:, None] + e_log[None, :]
            mask = np.triu(np.ones_like(score_mat))
            score_mat = score_mat * mask + (1 - mask) * -1e9

            idx = np.argmax(score_mat)
            best_start, best_end = np.unravel_index(idx, score_mat.shape)

            char_start = off[best_start][0]
            char_end = off[best_end][1]

            if char_start == 0 and char_end == 0:
                predictions.append(text)  # Fallback
            else:
                predictions.append(text[char_start:char_end])

    # Calculate Metric
    gt_selected = val_data["selected_texts"]
    scores = [jaccard(gt, pred) for pred, gt in zip(predictions, gt_selected)]
    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    df_val = pd.DataFrame(
        {
            "text": texts,
            "sentiment": sentiments,
            "selected_text": gt_selected,
            "prediction": predictions,
            "jaccard": scores,
            "text_len": [len(str(t)) for t in texts],
        }
    )

    df_val["error"] = 1.0 - df_val["jaccard"]
    corr = np.corrcoef(df_val["error"], df_val["text_len"])[0, 1]
    print(f"Correlation between Error and Text Length: {corr:.4f}")

    # 5. Submission
    if final_metric > 0.7093:
        print("\n=== Generating Submission ===")
        test_loader, test_ids = get_test_loader(load_cached_data=True)
        test_data = load_and_cache_data(Config.TEST_FILE, "test", load_cached_data=True)

        t_avg_start = np.zeros((len(test_ids), Config.MAX_LEN), dtype=np.float32)
        t_avg_end = np.zeros((len(test_ids), Config.MAX_LEN), dtype=np.float32)

        for fold, path in enumerate(model_paths):
            model = TweetModel()
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()

            _, s_log, e_log = eval_fn(test_loader, model, device)
            t_avg_start += s_log / Config.N_FOLDS
            t_avg_end += e_log / Config.N_FOLDS

            del model
            torch.cuda.empty_cache()
            gc.collect()

        sub_preds = []
        t_texts = test_data["texts"]
        t_sents = test_data["sentiments"]
        t_offs = test_data["offsets"]

        for i in range(len(test_ids)):
            text = t_texts[i]
            sentiment = t_sents[i]

            if sentiment == "neutral":
                sub_preds.append(text)
            else:
                s_log = t_avg_start[i]
                e_log = t_avg_end[i]
                off = t_offs[i]

                valid_mask = off[:, 1] > off[:, 0]
                s_log[~valid_mask] = -1e9
                e_log[~valid_mask] = -1e9

                score_mat = s_log[:, None] + e_log[None, :]
                mask = np.triu(np.ones_like(score_mat))
                score_mat = score_mat * mask + (1 - mask) * -1e9

                idx = np.argmax(score_mat)
                best_start, best_end = np.unravel_index(idx, score_mat.shape)

                char_start = off[best_start][0]
                char_end = off[best_end][1]

                if char_start == 0 and char_end == 0:
                    sub_preds.append(text)
                else:
                    sub_preds.append(text[char_start:char_end])

        sub_df = pd.DataFrame({"textID": test_ids, "selected_text": sub_preds})
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print("Validation metric too low. Skipping submission.")


if __name__ == "__main__":
    run()
