import os
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from library.config import Config
from library.data import get_loaders, get_test_loader
from library.model import TweetModel
from library.loss import SoftTargetKLLoss, RDropLoss
from library.engine import train_fn, eval_fn
from library.utils import seed_everything, get_optimizer_params, jaccard, normalize_text


def predict_loader(model, loader, device):
    """
    Runs inference on a loader and returns start/end probabilities.
    """
    model.eval()
    start_probs_list = []
    end_probs_list = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            start_probs_list.append(start_probs)
            end_probs_list.append(end_probs)

    return np.concatenate(start_probs_list), np.concatenate(end_probs_list)


def run():
    # 1. Setup
    device = Config.device
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading Tokenizer and Data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    train_loader, val_loader = get_loaders(tokenizer, load_cached_data=True)

    # 3. Training Loop (Ensemble of 2 Seeds)
    seeds = [42, 43]
    model_paths = []

    # Losses
    criterion_task = SoftTargetKLLoss()
    criterion_rdrop = RDropLoss()

    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---")
        seed_everything(seed)

        model = TweetModel(Config)
        model.to(device)

        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=Config.lr,
            decoder_lr=Config.lr * 5,
            weight_decay=Config.weight_decay,
            llrd_decay=Config.llrd_decay,
        )
        optimizer = AdamW(optimizer_parameters, lr=Config.lr, eps=Config.eps)

        num_epochs = Config.epochs
        num_train_steps = len(train_loader) * num_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        best_jaccard = 0.0
        save_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.bin")

        for epoch in range(num_epochs):
            train_loss = train_fn(
                train_loader,
                model,
                optimizer,
                device,
                scheduler,
                criterion_task,
                criterion_rdrop,
                Config,
            )
            val_loss, val_jaccard = eval_fn(
                val_loader, model, device, criterion_task, Config
            )

            print(
                f"Seed {seed} | Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
            )

            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved for seed {seed}")

        model_paths.append(save_path)

    # 4. Ensemble Validation
    print("\nComputing Ensemble Validation Metric...")
    # Load validation metadata
    val_meta_df = pd.read_csv(Config.VAL_META_PATH)
    # Non-neutral subset for prediction
    non_neutral_df = val_meta_df[val_meta_df["sentiment"] != "neutral"].reset_index(
        drop=True
    )

    # Accumulate probabilities
    avg_start_probs = None
    avg_end_probs = None

    for path in model_paths:
        model = TweetModel(Config)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)

        s_probs, e_probs = predict_loader(model, val_loader, device)

        if avg_start_probs is None:
            avg_start_probs = s_probs
            avg_end_probs = e_probs
        else:
            avg_start_probs += s_probs
            avg_end_probs += e_probs

    avg_start_probs /= len(model_paths)
    avg_end_probs /= len(model_paths)

    # Decode and Compute Metric
    non_neutral_jaccards = []

    # We need offsets and texts. Since loader is sequential, we can iterate it again or cache them.
    # To be safe, let's iterate loader to get auxiliary data.
    # Note: predict_loader iterated sequentially.

    current_idx = 0
    for batch in val_loader:
        offsets = batch["offsets"].numpy()
        texts = batch["text"]
        batch_size = len(texts)

        for i in range(batch_size):
            start_p = avg_start_probs[current_idx]
            end_p = avg_end_probs[current_idx]
            offset = offsets[i]
            original_text = texts[i]
            target_text = str(non_neutral_df.iloc[current_idx]["selected_text"])

            # Decode
            score_mat = np.outer(start_p, end_p)
            score_mat = np.triu(score_mat)
            best_idx = np.argmax(score_mat)
            best_start_idx, best_end_idx = np.unravel_index(best_idx, score_mat.shape)

            if best_start_idx >= len(offset) or best_end_idx >= len(offset):
                pred_text = original_text
            else:
                char_start = offset[best_start_idx][0]
                char_end = offset[best_end_idx][1]
                if char_start == 0 and char_end == 0:
                    pred_text = original_text
                else:
                    pred_text = original_text[char_start:char_end]

            non_neutral_jaccards.append(jaccard(target_text, pred_text))
            current_idx += 1

    avg_non_neutral_jaccard = np.mean(non_neutral_jaccards)

    # Calculate Neutral Jaccard
    neutral_df = val_meta_df[val_meta_df["sentiment"] == "neutral"]
    neutral_jaccards = [
        jaccard(str(r.text), str(r.selected_text)) for r in neutral_df.itertuples()
    ]
    avg_neutral_jaccard = np.mean(neutral_jaccards) if neutral_jaccards else 0.0

    n_neutral = len(neutral_df)
    n_non_neutral = len(non_neutral_df)
    total_samples = n_neutral + n_non_neutral

    final_metric = (
        avg_neutral_jaccard * n_neutral + avg_non_neutral_jaccard * n_non_neutral
    ) / total_samples

    print(f"Neutral Jaccard: {avg_neutral_jaccard:.4f}")
    print(f"Non-Neutral Jaccard: {avg_non_neutral_jaccard:.4f}")
    print(f"Final Ensemble Validation Metric: {final_metric}")

    # 5. Failure Analysis (using ensemble predictions)
    val_errors = [1.0 - s for s in non_neutral_jaccards]
    val_text_lens = [len(str(t)) for t in non_neutral_df["text"]]
    if val_errors:
        corr = np.corrcoef(val_errors, val_text_lens)[0, 1]
        print(f"Correlation between Error and Text Length: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.7093
    if final_metric > THRESHOLD:
        print(f"\nMetric {final_metric:.4f} > {THRESHOLD}. Generating submission...")
        test_loader = get_test_loader(tokenizer)

        # Ensemble Prediction on Test
        avg_start_probs = None
        avg_end_probs = None

        for path in model_paths:
            model = TweetModel(Config)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            s_probs, e_probs = predict_loader(model, test_loader, device)

            if avg_start_probs is None:
                avg_start_probs = s_probs
                avg_end_probs = e_probs
            else:
                avg_start_probs += s_probs
                avg_end_probs += e_probs

        avg_start_probs /= len(model_paths)
        avg_end_probs /= len(model_paths)

        # Decode
        test_df = pd.read_csv(Config.TEST_META_PATH)
        predictions = []
        ids = []
        current_idx = 0

        # Iterate loader to get offsets/texts
        for batch in test_loader:
            offsets = batch["offsets"].numpy()
            texts = batch["text"]
            sentiments = batch["sentiment"]
            batch_size = len(texts)

            for i in range(batch_size):
                text_id = test_df.iloc[current_idx]["textID"]
                sentiment = sentiments[i]
                original_text = texts[i]
                offset = offsets[i]

                if sentiment == "neutral":
                    pred_text = original_text
                else:
                    start_p = avg_start_probs[current_idx]
                    end_p = avg_end_probs[current_idx]
                    score_mat = np.outer(start_p, end_p)
                    score_mat = np.triu(score_mat)
                    best_idx = np.argmax(score_mat)
                    best_start_idx, best_end_idx = np.unravel_index(
                        best_idx, score_mat.shape
                    )

                    if best_start_idx >= len(offset) or best_end_idx >= len(offset):
                        pred_text = original_text
                    else:
                        char_start = offset[best_start_idx][0]
                        char_end = offset[best_end_idx][1]
                        if char_start == 0 and char_end == 0:
                            pred_text = original_text
                        else:
                            pred_text = original_text[char_start:char_end]

                predictions.append(f'"{pred_text}"')
                ids.append(text_id)
                current_idx += 1

        sub_df = pd.DataFrame({"textID": ids, "selected_text": predictions})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"Metric {final_metric:.4f} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run()
