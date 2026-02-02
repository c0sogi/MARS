import os
import gc
import csv
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold

# Import library modules
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import (
    process_data,
    TweetDataset,
    SmartBatchingCollate,
    LengthSortedBatchSampler,
)
from library.model import TweetModel
from library.engine import train_fn


def decode_prediction(start_probs, end_probs, orig_text, sentiment, offsets):
    """
    Decodes start and end probabilities into a string prediction.
    Applies the neutral heuristic and ensures valid span selection.
    """
    if sentiment == "neutral":
        return orig_text

    # Compute joint probability matrix (start, end)
    # shape: (seq_len, seq_len)
    score_matrix = np.outer(start_probs, end_probs)

    # Enforce start <= end by zeroing out the lower triangle (where col < row)
    # np.triu keeps the upper triangle including diagonal
    score_matrix = np.triu(score_matrix)

    # Find best indices
    best_idx = np.argmax(score_matrix)
    start_idx, end_idx = np.unravel_index(best_idx, score_matrix.shape)

    # Map to characters
    if start_idx >= len(offsets) or end_idx >= len(offsets):
        return orig_text

    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]

    # Handle case where special tokens (0,0) are selected or empty span
    if char_start == 0 and char_end == 0:
        return orig_text

    return orig_text[char_start:char_end]


def inference(model_paths, dataloader, device, config, is_test=False):
    """
    Performs ensemble inference.
    Sums probabilities from all models to reduce variance.
    """
    num_samples = len(dataloader.dataset)

    # Accumulators for probabilities
    # Using CPU tensors to aggregate to save GPU memory
    all_start_probs = torch.zeros((num_samples, config.max_len), dtype=torch.float32)
    all_end_probs = torch.zeros((num_samples, config.max_len), dtype=torch.float32)

    # Iterate through each trained model in the ensemble
    for model_path in model_paths:
        # Load Model
        model = TweetModel(config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        current_idx = 0

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc=f"Inf {os.path.basename(model_path)}", leave=False
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)

                # Forward pass
                start_logits, end_logits = model(
                    input_ids, attention_mask, token_type_ids
                )

                # Convert to probabilities
                start_probs = torch.softmax(start_logits, dim=1).cpu()
                end_probs = torch.softmax(end_logits, dim=1).cpu()

                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)

                # Pad to max_len to allow stacking
                padded_start = torch.zeros(
                    (batch_size, config.max_len), dtype=torch.float32
                )
                padded_end = torch.zeros(
                    (batch_size, config.max_len), dtype=torch.float32
                )

                padded_start[:, :seq_len] = start_probs
                padded_end[:, :seq_len] = end_probs

                # Accumulate
                all_start_probs[current_idx : current_idx + batch_size] += padded_start
                all_end_probs[current_idx : current_idx + batch_size] += padded_end

                current_idx += batch_size

        # Cleanup to free GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Decode predictions
    predictions = []
    scores = []

    all_start_probs = all_start_probs.numpy()
    all_end_probs = all_end_probs.numpy()

    current_idx = 0
    # Iterate dataloader again to retrieve metadata for decoding
    for batch in tqdm(dataloader, desc="Decoding", leave=False):
        orig_texts = batch["orig_text"]
        sentiments = batch["sentiment"]
        offsets = batch["offsets"].numpy()
        selected_texts = batch["selected_text"]

        batch_size = len(orig_texts)

        start_p_batch = all_start_probs[current_idx : current_idx + batch_size]
        end_p_batch = all_end_probs[current_idx : current_idx + batch_size]

        for i in range(batch_size):
            pred = decode_prediction(
                start_p_batch[i],
                end_p_batch[i],
                orig_texts[i],
                sentiments[i],
                offsets[i],
            )
            predictions.append(pred)

            if not is_test:
                scores.append(jaccard(selected_texts[i], pred))

        current_idx += batch_size

    if is_test:
        return 0.0, predictions
    else:
        return np.mean(scores), predictions


def main():
    # 1. Configuration
    config = Config()
    # Override defaults for fast baseline execution
    config.epochs = 2
    config.train_batch_size = 16
    config.valid_batch_size = 64

    seed_everything(config.seed)

    print("Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.train_batch_size}")

    # 2. Data Loading
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_meta = pd.read_csv(config.train_path)
    val_meta = pd.read_csv(config.val_path)
    test_meta = pd.read_csv(config.test_path)
    test_meta["selected_text"] = ""  # Placeholder for processing

    # Process Hold-out Validation Set
    print("\nProcessing Hold-out Validation Set...")
    val_data = process_data(val_meta, tokenizer, config, mode="holdout_val")
    val_dataset = TweetDataset(*val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        collate_fn=SmartBatchingCollate(),
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Process Test Set
    print("\nProcessing Test Set...")
    test_data = process_data(test_meta, tokenizer, config, mode="test")
    test_dataset = TweetDataset(*test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        collate_fn=SmartBatchingCollate(),
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 3. Training (Stratified K-Fold)
    print("\nPreparing Training Data (Full)...")
    # Load and process the full training set once
    full_train_data = process_data(train_meta, tokenizer, config, mode="train_full")
    (
        input_ids_all,
        attention_masks_all,
        token_type_ids_all,
        start_labels_all,
        end_labels_all,
        offsets_all,
        orig_texts_all,
        sentiments_all,
        selected_texts_all,
    ) = full_train_data

    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )

    model_paths = []

    for fold, (train_idx, _) in enumerate(
        skf.split(train_meta, train_meta["sentiment"])
    ):
        print(f"\n{'#'*30}")
        print(f"Training Fold {fold+1}/{config.n_folds}")
        print(f"{'#'*30}")

        # Create Dataset for the current fold
        train_ds = TweetDataset(
            input_ids_all[train_idx],
            attention_masks_all[train_idx],
            token_type_ids_all[train_idx],
            start_labels_all[train_idx],
            end_labels_all[train_idx],
            offsets_all[train_idx],
            orig_texts_all[train_idx],
            sentiments_all[train_idx],
            selected_texts_all[train_idx],
        )

        # Use LengthSortedBatchSampler for efficiency
        train_sampler = LengthSortedBatchSampler(
            train_ds, batch_size=config.train_batch_size, drop_last=True, shuffle=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=SmartBatchingCollate(),
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Initialize Model and Optimizer
        model = TweetModel(config)
        model.to(config.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        num_train_steps = int(len(train_ds) / config.train_batch_size * config.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_train_steps, eta_min=1e-6
        )

        # Training Loop
        for epoch in range(config.epochs):
            loss = train_fn(
                train_loader, model, optimizer, config.device, scheduler, config, epoch
            )
            print(f"Fold {fold+1} Epoch {epoch+1}: Loss = {loss:.4f}")

        # Save Model
        model_name = f"model_fold_{fold}.pth"
        save_path = os.path.join(config.model_dir, model_name)
        torch.save(model.state_dict(), save_path)
        model_paths.append(save_path)
        print(f"Saved model to {save_path}")

        # Cleanup
        del model, optimizer, scheduler, train_loader, train_ds
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Final Validation
    print("\nRunning Ensemble Validation on Hold-out Set...")
    val_score, val_preds = inference(
        model_paths, val_loader, config.device, config, is_test=False
    )
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    val_meta["pred"] = val_preds
    val_meta["jaccard"] = val_meta.apply(
        lambda x: jaccard(x["selected_text"], x["pred"]), axis=1
    )
    val_meta["error"] = 1.0 - val_meta["jaccard"]
    val_meta["text_len"] = val_meta["text"].apply(lambda x: len(str(x).split()))

    corr = val_meta["error"].corr(val_meta["text_len"])
    print(f"Correlation (Error vs Text Length): {corr:.4f}")

    print("Mean Jaccard by Sentiment:")
    print(val_meta.groupby("sentiment")["jaccard"].mean())

    # 6. Submission
    if val_score > 0.7205:
        print("\nMetric > 0.7205. Generating Submission...")
        _, test_preds = inference(
            model_paths, test_loader, config.device, config, is_test=True
        )

        sub_df = pd.DataFrame(
            {"textID": test_meta["textID"], "selected_text": test_preds}
        )

        # Save with quotes as required by the format
        sub_df.to_csv(config.submission_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Submission saved to {config.submission_path}")
    else:
        print(f"\nMetric {val_score} <= 0.7205. Submission skipped.")


if __name__ == "__main__":
    main()
