import os
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from library.config import Config
from library.data import get_loaders, get_test_loader
from library.model import TweetModel
from library.loss import SoftTargetKLLoss
from library.engine import train_fn, eval_fn
from library.utils import seed_everything, get_optimizer_params, jaccard, normalize_text


def predict_probs(model, loader, device):
    """
    Runs inference on a loader and returns start/end probabilities.
    """
    model.eval()
    all_start_probs = []
    all_end_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            all_start_probs.append(start_probs)
            all_end_probs.append(end_probs)

    return np.concatenate(all_start_probs), np.concatenate(all_end_probs)


def run():
    # 1. Setup
    device = Config.device
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading Tokenizer and Data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    train_loader, val_loader = get_loaders(tokenizer, load_cached_data=True)

    # 3. Training Loop (Seed Averaging Ensemble)
    model_paths = []

    for seed in Config.seeds:
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

        criterion_task = SoftTargetKLLoss()

        best_jaccard = 0.0
        seed_model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.bin")

        for epoch in range(num_epochs):
            train_loss = train_fn(
                train_loader,
                model,
                optimizer,
                device,
                scheduler,
                criterion_task,
                Config,
            )
            val_loss, val_jaccard = eval_fn(
                val_loader, model, device, criterion_task, Config
            )
            print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
            )

            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                torch.save(model.state_dict(), seed_model_path)
                print(f"New best model for seed {seed} saved.")

        model_paths.append(seed_model_path)
        # Free memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # 4. Ensemble Validation
    print("\nComputing Ensemble Validation Metric...")

    # Accumulate probabilities
    ensemble_start_probs = None
    ensemble_end_probs = None

    for path in model_paths:
        model = TweetModel(Config)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)

        s_probs, e_probs = predict_probs(model, val_loader, device)

        if ensemble_start_probs is None:
            ensemble_start_probs = s_probs
            ensemble_end_probs = e_probs
        else:
            ensemble_start_probs += s_probs
            ensemble_end_probs += e_probs

        del model
        torch.cuda.empty_cache()

    # Average
    ensemble_start_probs /= len(Config.seeds)
    ensemble_end_probs /= len(Config.seeds)

    # Decode and Compute Metric
    val_meta_df = pd.read_csv(Config.VAL_META_PATH)
    neutral_df = val_meta_df[val_meta_df["sentiment"] == "neutral"]
    non_neutral_df = val_meta_df[val_meta_df["sentiment"] != "neutral"].reset_index(
        drop=True
    )

    # Calculate Neutral Jaccard
    neutral_jaccards = [
        jaccard(str(row["text"]), str(row["selected_text"]))
        for _, row in neutral_df.iterrows()
    ]
    avg_neutral_jaccard = np.mean(neutral_jaccards) if neutral_jaccards else 0.0

    # Calculate Non-Neutral Jaccard from Ensemble Predictions
    non_neutral_jaccards = []
    val_errors = []
    val_text_lens = []

    # Iterate through validation samples (aligned with val_loader)
    # val_loader contains only non-neutrals
    batch_idx = 0
    sample_idx = 0

    # We need to iterate over the loader again just to get offsets and texts
    # or we can assume order is preserved (it is) and use the dataframe + cached offsets if available
    # Safest is to iterate loader to get auxiliary data

    current_idx = 0
    for batch in val_loader:
        offsets = batch["offsets"].numpy()
        texts = batch["text"]
        batch_size = len(texts)

        for i in range(batch_size):
            start_p = ensemble_start_probs[current_idx]
            end_p = ensemble_end_probs[current_idx]
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

            score = jaccard(target_text, pred_text)
            non_neutral_jaccards.append(score)

            val_errors.append(1.0 - score)
            val_text_lens.append(len(str(original_text)))

            current_idx += 1

    avg_non_neutral_jaccard = np.mean(non_neutral_jaccards)

    final_metric = (
        avg_neutral_jaccard * len(neutral_df)
        + avg_non_neutral_jaccard * len(non_neutral_df)
    ) / len(val_meta_df)

    print(f"Neutral Jaccard: {avg_neutral_jaccard:.4f}")
    print(f"Non-Neutral Jaccard: {avg_non_neutral_jaccard:.4f}")
    print(f"Final Validation Metric: {final_metric:.4f}")

    # Failure Analysis
    if len(val_errors) > 0:
        correlation = np.corrcoef(val_errors, val_text_lens)[0, 1]
        print(f"\nCorrelation between Error and Text Length: {correlation:.4f}")

    # 5. Submission
    THRESHOLD = 0.7093
    if final_metric > THRESHOLD:
        print(f"\nMetric > {THRESHOLD}. Generating submission...")
        test_loader = get_test_loader(tokenizer)
        test_df = pd.read_csv(Config.TEST_META_PATH)

        # Ensemble Inference on Test
        ens_test_start = None
        ens_test_end = None

        for path in model_paths:
            model = TweetModel(Config)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            s_probs, e_probs = predict_probs(model, test_loader, device)

            if ens_test_start is None:
                ens_test_start = s_probs
                ens_test_end = e_probs
            else:
                ens_test_start += s_probs
                ens_test_end += e_probs
            del model
            torch.cuda.empty_cache()

        ens_test_start /= len(Config.seeds)
        ens_test_end /= len(Config.seeds)

        # Decode Test
        predictions = []
        ids = []
        row_idx = 0

        # Iterate loader for aux data
        for batch in test_loader:
            offsets = batch["offsets"].numpy()
            texts = batch["text"]
            sentiments = batch["sentiment"]
            batch_size = len(texts)

            for i in range(batch_size):
                text_id = test_df.iloc[row_idx]["textID"]
                sentiment = sentiments[i]
                original_text = texts[i]
                offset = offsets[i]

                if sentiment == "neutral":
                    pred_text = original_text
                else:
                    start_p = ens_test_start[row_idx]
                    end_p = ens_test_end[row_idx]

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
                row_idx += 1

        sub_df = pd.DataFrame({"textID": ids, "selected_text": predictions})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run()
