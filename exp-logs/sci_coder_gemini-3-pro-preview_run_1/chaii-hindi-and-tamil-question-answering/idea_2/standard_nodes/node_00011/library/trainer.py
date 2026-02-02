import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, compute_score
from library.model_handler import get_model, get_tokenizer, save_model
from library.data_loader import get_processed_data, QADataset
from library.post_processor import postprocess_qa_predictions, save_submission


def train_fn(model, data_loader, optimizer, scheduler, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    # Iterate over batches
    # Using simple print for progress to avoid tqdm clutter in logs if not desired
    for batch_idx, batch in enumerate(data_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # AutoModelForQuestionAnswering computes loss if labels are provided
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimizer and Scheduler step
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.6f}")
    return avg_loss


def eval_fn(model, data_loader, raw_examples, features_df, device):
    """
    Evaluates the model on the validation set using Jaccard score.
    """
    model.eval()

    all_start_logits = []
    all_end_logits = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass (no labels needed for inference)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Collect logits -> move to CPU -> numpy
            start_logits = outputs.start_logits.detach().cpu().numpy()
            end_logits = outputs.end_logits.detach().cpu().numpy()

            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

    # Concatenate all batches
    # Shape: (num_features, seq_len)
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # Post-process to get text predictions
    # This handles the sliding window aggregation
    predictions = postprocess_qa_predictions(
        examples=raw_examples,
        features=features_df,
        raw_predictions=(all_start_logits, all_end_logits),
    )

    # Calculate Jaccard Score
    # Map predictions to ground truth
    # raw_examples has 'id' and 'answer_text'
    ground_truths = []
    preds_list = []

    for _, row in raw_examples.iterrows():
        ex_id = row["id"]
        gt_text = str(row["answer_text"])
        pred_text = predictions.get(ex_id, "")

        ground_truths.append(gt_text)
        preds_list.append(pred_text)

    score = compute_score(ground_truths, preds_list)
    print(f"Validation Jaccard Score: {score}")

    return score


def predict_fn(model, data_loader, raw_examples, features_df, device):
    """
    Generates predictions for the test set.
    """
    model.eval()

    all_start_logits = []
    all_end_logits = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            start_logits = outputs.start_logits.detach().cpu().numpy()
            end_logits = outputs.end_logits.detach().cpu().numpy()

            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    predictions = postprocess_qa_predictions(
        examples=raw_examples,
        features=features_df,
        raw_predictions=(all_start_logits, all_end_logits),
    )

    return predictions


def run_training():
    """
    Main orchestration function.
    """
    # 1. Setup
    set_seed(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # 2. Data Preparation
    tokenizer = get_tokenizer()

    # Load processed features (sliding windows)
    # This handles caching internally
    train_features, val_features, test_features = get_processed_data(
        tokenizer, load_cached_data=True
    )

    # Load raw examples (needed for post-processing and metrics)
    # We read directly from metadata paths defined in Config
    if Config.debug:
        raw_val = pd.read_csv(Config.val_path).head(Config.debug_subset_size)
        raw_test = pd.read_csv(Config.test_path).head(Config.debug_subset_size)
    else:
        raw_val = pd.read_csv(Config.val_path)
        raw_test = pd.read_csv(Config.test_path)

    # Create Datasets
    train_dataset = QADataset(train_features, mode="train")
    val_dataset = QADataset(val_features, mode="val")
    test_dataset = QADataset(test_features, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.eval_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.eval_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = get_model()  # Loads pre-trained XLM-R

    # Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Scheduler
    num_train_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # 4. Training Loop
    best_score = -1.0

    print("Starting training...")
    for epoch in range(Config.epochs):
        # Train
        train_fn(model, train_loader, optimizer, scheduler, device, epoch)

        # Validate
        val_score = eval_fn(model, val_loader, raw_val, val_features, device)

        # Early Stopping / Save Best
        if val_score > best_score:
            print(f"Score improved from {best_score} to {val_score}. Saving model...")
            best_score = val_score
            save_model(model, tokenizer, Config.model_output_dir)
        else:
            print(f"Score {val_score} did not improve best {best_score}.")

    print(f"Training complete. Best Validation Score: {best_score}")

    # 5. Inference on Test Set
    print("Loading best model for inference...")
    # Load the best weights saved during training
    best_model = get_model(weights_path=Config.model_output_dir)

    print("Generating predictions on test set...")
    test_predictions = predict_fn(
        best_model, test_loader, raw_test, test_features, device
    )

    # 6. Submission
    print(f"Saving submission to {Config.submission_path}...")
    save_submission(test_predictions, Config.submission_path)
    print("Done.")
