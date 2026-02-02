import os
import gc
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.utils import seed_everything, jaccard
from library.dataset import get_data, create_loader, TweetDataset, SmartBatchingCollate
from library.model import TweetModel
from library.engine import train_fn, eval_fn


def run_training():
    """
    Orchestrates the 5-Fold Stratified Cross-Validation training.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.OUTPUT_MODEL_DIR, exist_ok=True)

    # Override Config for speed and hardware optimization within 2h limit
    Config.EPOCHS = 2  # Sufficient for fine-tuning DeBERTa-Large
    Config.TRAIN_BATCH_SIZE = 16  # Increase batch size for A100 (40GB)

    print(
        f"Starting training with {Config.N_FOLDS} folds, {Config.EPOCHS} epochs each."
    )
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Load full training data from metadata
    # We use the metadata/train.csv as the base for CV
    train_data_raw = get_data(
        Config.TRAINING_FILE,
        tokenizer,
        Config.MAX_LEN,
        Config.CACHE_DIR,
        load_cached_data=True,
    )

    # Prepare for Stratified Split
    sentiments = [x["sentiment"] for x in train_data_raw]
    kf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # 3. Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data_raw, sentiments)):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Subset data
        train_subset = [train_data_raw[i] for i in train_idx]
        val_subset = [train_data_raw[i] for i in val_idx]

        # Create Loaders
        train_loader = create_loader(
            train_subset, tokenizer, Config.TRAIN_BATCH_SIZE, is_train=True
        )
        val_loader = create_loader(
            val_subset, tokenizer, Config.VALID_BATCH_SIZE, is_train=False
        )

        # Initialize Model
        model = TweetModel(Config.MODEL_PATH)
        model.to(Config.DEVICE)

        # Optimizer & Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        num_train_steps = int(
            len(train_subset) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        # Training Epochs
        best_jaccard = 0
        model_save_path = os.path.join(
            Config.OUTPUT_MODEL_DIR, f"model_fold_{fold}.pth"
        )

        for epoch in range(Config.EPOCHS):
            train_loss = train_fn(
                train_loader, model, optimizer, Config.DEVICE, scheduler
            )
            val_loss, val_jaccard = eval_fn(val_loader, model, Config.DEVICE)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
            )

            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved Best Model for Fold {fold+1}")

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()


def inference_ensemble(data_path, tokenizer, model_paths):
    """
    Performs ensemble inference on the provided dataset.
    Returns a DataFrame with predictions.
    """
    # Load Data
    data = get_data(
        data_path, tokenizer, Config.MAX_LEN, Config.CACHE_DIR, load_cached_data=True
    )
    dataset = TweetDataset(data)
    collate = SmartBatchingCollate(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=Config.VALID_BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Logits Containers
    # shape: (num_samples, max_len) - we'll accumulate probabilities or logits here
    # Since sequence lengths vary, we process batch by batch and store results

    # To handle variable lengths in ensemble, we iterate through the loader once for each model
    # and accumulate logits. However, simpler is to load all models (if memory permits) or
    # iterate models and accumulate. Given A100 40GB, we can load 1 model at a time and accumulate.

    # We need to map back to original indices to ensure order
    # The loader is sequential (shuffle=False), so order is preserved.

    num_samples = len(data)
    # We will store final start/end probabilities for each sample
    # Since lengths differ, we store in a list
    all_start_probs = [None] * num_samples
    all_end_probs = [None] * num_samples

    print(f"Running ensemble inference on {data_path}...")

    for model_path in model_paths:
        print(f"Loading {os.path.basename(model_path)}...")
        model = TweetModel(Config.MODEL_PATH)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        model.eval()

        current_idx = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(Config.DEVICE)
                mask = batch["attention_mask"].to(Config.DEVICE)
                token_type_ids = batch["token_type_ids"].to(Config.DEVICE)

                # Forward
                start_logits, end_logits = model(input_ids, mask, token_type_ids)

                # Softmax to get probs
                start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
                end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

                batch_size = input_ids.size(0)

                for i in range(batch_size):
                    global_idx = current_idx + i

                    # Initialize if first model
                    if all_start_probs[global_idx] is None:
                        all_start_probs[global_idx] = start_probs[i]
                        all_end_probs[global_idx] = end_probs[i]
                    else:
                        # Add to existing. Note: lengths might differ due to dynamic padding in different runs?
                        # No, we use the SAME loader for all models, so batches and padding are identical
                        # because we re-create the loader or iterate it deterministically.
                        # Wait, SmartBatchingSampler shuffles. We used shuffle=False in this loader.
                        # SmartBatchingCollate pads to batch max.
                        # Since we iterate the loader inside the model loop, we must ensure the loader yields identical batches.
                        # Standard DataLoader with shuffle=False is deterministic.

                        # However, to be safe against shape mismatch if we re-instantiated loader (we didn't),
                        # we just add.

                        # Check shape compatibility (just in case)
                        if all_start_probs[global_idx].shape != start_probs[i].shape:
                            # This should not happen if loader is reused or deterministic
                            raise ValueError("Shape mismatch in ensemble")

                        all_start_probs[global_idx] += start_probs[i]
                        all_end_probs[global_idx] += end_probs[i]

                current_idx += batch_size

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Decode Predictions
    predictions = []
    print("Decoding predictions...")

    for i in range(num_samples):
        item = data[i]
        text = item["text"]
        sentiment = item["sentiment"]
        offsets = item["offsets"]

        # Neutral Heuristic
        if sentiment == "neutral":
            pred_text = text
        else:
            # Average probs
            start_p = all_start_probs[i]
            end_p = all_end_probs[i]

            idx_start = np.argmax(start_p)
            idx_end = np.argmax(end_p)

            if idx_end < idx_start:
                idx_end = idx_start

            # Clip to offset bounds
            if idx_start >= len(offsets):
                idx_start = len(offsets) - 1
            if idx_end >= len(offsets):
                idx_end = len(offsets) - 1

            start_char = offsets[idx_start][0]
            end_char = offsets[idx_end][1]

            pred_text = text[start_char:end_char]

        predictions.append(
            {
                "textID": item.get(
                    "textID", ""
                ),  # Assuming textID is preserved in get_data if available, else need to handle
                "selected_text": pred_text,
            }
        )

    # Re-attach textIDs correctly.
    # get_data processes raw csv. We need to ensure we map back to textID.
    # The process_data function in dataset.py does NOT store textID.
    # We need to fetch textIDs from the source CSV to align.

    df_source = pd.read_csv(data_path)
    # Filter same way as get_data
    if "selected_text" in df_source.columns:
        df_source = df_source.dropna(subset=["text", "selected_text", "sentiment"])
    else:
        df_source = df_source.dropna(subset=["text", "sentiment"])

    # Handle Debug logic from get_data
    if Config.DEBUG:
        df_source = df_source.head(Config.DEBUG_SAMPLE_SIZE)

    # Assign textIDs
    final_preds = pd.DataFrame(predictions)
    final_preds["textID"] = df_source["textID"].values

    return final_preds


def failure_analysis(val_preds, val_path):
    """
    Analyzes failure modes on the validation set.
    """
    print("\n--- Failure Analysis ---")
    df_val = pd.read_csv(val_path)
    df_val = df_val.dropna(subset=["text", "selected_text", "sentiment"])

    # Merge predictions
    # Ensure alignment
    if len(df_val) != len(val_preds):
        print(
            f"Warning: Validation set size ({len(df_val)}) != Prediction size ({len(val_preds)}). Alignment might be off."
        )
        # This happens if get_data drops rows that pd.read_csv keeps, or vice versa.
        # We used the same dropna logic.

    # Merge on textID
    merged = pd.merge(df_val, val_preds, on="textID", suffixes=("_true", "_pred"))

    # Calculate Jaccard
    merged["jaccard"] = merged.apply(
        lambda x: jaccard(x["selected_text_pred"], x["selected_text_true"]), axis=1
    )
    merged["error"] = 1.0 - merged["jaccard"]

    # Feature: Text Length
    merged["text_len"] = merged["text"].apply(lambda x: len(str(x).split()))

    # Correlation
    corr = merged["error"].corr(merged["text_len"])
    print(f"Correlation (Error vs Input Length): {corr:.4f}")

    # Error by Sentiment
    print("Mean Jaccard by Sentiment:")
    print(merged.groupby("sentiment")["jaccard"].mean())

    return merged["jaccard"].mean()


def main():
    # 1. Train
    run_training()

    # 2. Validation Inference (Ensemble)
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)
    model_paths = [
        os.path.join(Config.OUTPUT_MODEL_DIR, f"model_fold_{i}.pth")
        for i in range(Config.N_FOLDS)
    ]

    print("\nRunning Validation Inference...")
    val_preds = inference_ensemble(Config.VALIDATION_FILE, tokenizer, model_paths)

    # 3. Failure Analysis & Metric
    final_metric = failure_analysis(val_preds, Config.VALIDATION_FILE)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # 4. Submission
    THRESHOLD = 0.7205
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.4f}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        test_preds = inference_ensemble(Config.TEST_FILE, tokenizer, model_paths)

        # Format for submission
        submission = test_preds[["textID", "selected_text"]]
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric:.4f}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
