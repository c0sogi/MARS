import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Import provided library components
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_f1_score,
    save_checkpoint,
    load_checkpoint,
)
from library.data_processing import prepare_data, StackExchangeDataset
from library.model import TextCNN
from library.engine import train_fn, eval_fn, inference_fn


def main():
    # 1. Initialization
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Override Config for Fast Baseline
    # Limit epochs to ensure execution completes well within 2 hours
    Config.EPOCHS = 5

    # 2. Data Loading
    print("Loading data...")
    # Load cached data to save time
    (train_data, val_data, test_data, tokenizer, mlb) = prepare_data(
        load_cached_data=True
    )

    train_tokens, train_labels = train_data
    val_tokens, val_labels = val_data
    test_tokens, test_ids = test_data

    # Subsample training data for speed (Fast Baseline requirement)
    # 500,000 samples is sufficient for a strong baseline while being fast to train
    MAX_TRAIN_SAMPLES = 500000
    if len(train_tokens) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(train_tokens)} to {MAX_TRAIN_SAMPLES} samples..."
        )
        indices = np.random.choice(len(train_tokens), MAX_TRAIN_SAMPLES, replace=False)
        train_tokens = train_tokens[indices]
        train_labels = train_labels[indices]

    # Create Datasets
    train_dataset = StackExchangeDataset(train_tokens, train_labels)
    val_dataset = StackExchangeDataset(val_tokens, val_labels)
    test_dataset = StackExchangeDataset(test_tokens, ids=test_ids)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = TextCNN(
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_classes=len(mlb.classes_),
        kernel_sizes=Config.KERNEL_SIZES,
        num_filters=Config.NUM_FILTERS,
        dropout=Config.DROPOUT,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 4. Training Loop
    print("Starting training...")
    best_val_f1 = 0.0

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = eval_fn(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val F1: {val_f1:.5f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(model, optimizer, epoch, val_loss, Config.MODEL_SAVE_PATH)

    # 5. Validation Assessment & Failure Analysis
    print("Loading best model for validation assessment...")
    load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)
    model.eval()

    # Run inference on full validation set
    val_probs_list = []
    val_labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            tokens = batch["tokens"].to(device)
            labels = batch["labels"].to(device)

            logits = model(tokens)
            probs = torch.sigmoid(logits)

            val_probs_list.append(probs.cpu().numpy())
            val_labels_list.append(labels.cpu().numpy())

    val_probs = np.vstack(val_probs_list)
    val_true = np.vstack(val_labels_list)
    val_preds = (val_probs > 0.5).astype(int)

    # Compute Final Metric
    final_metric = calculate_f1_score(val_true, val_preds, average="micro")
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Input Length
    print("Performing failure analysis...")
    # Calculate sample-wise F1 to determine error per sample
    sample_f1 = calculate_f1_score(val_true, val_preds, average="samples")
    errors = 1.0 - sample_f1

    # Calculate input length (number of non-padding tokens)
    # val_tokens corresponds to the order of val_loader (shuffle=False)
    lengths = np.sum(val_tokens != 0, axis=1)

    # Compute correlation
    if len(errors) > 1:
        correlation = np.corrcoef(errors, lengths)[0, 1]
    else:
        correlation = 0.0

    print(f"Correlation between Error and Input Length: {correlation:.6f}")

    # 6. Submission Generation
    print("Generating submission...")
    test_probs, test_ids_out = inference_fn(model, test_loader, device)
    test_preds = (test_probs > 0.5).astype(int)

    # Convert binary predictions back to tag strings
    pred_tags_tuples = mlb.inverse_transform(test_preds)
    pred_tags_list = [" ".join(tags) for tags in pred_tags_tuples]

    # Create submission DataFrame
    df_submission = pd.DataFrame({"Id": test_ids_out, "Tags": pred_tags_list})

    # Save to disk
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
