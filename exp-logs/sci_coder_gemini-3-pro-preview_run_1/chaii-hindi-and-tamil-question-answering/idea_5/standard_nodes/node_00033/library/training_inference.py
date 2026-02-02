import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import collections
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Import from provided library
from library.configuration import Config
from library.utilities import set_seed
from library.data_processing import get_train_data, get_test_data
from library.modeling import MultiTaskQAModel


def train_fn(config):
    """
    Trains the MultiTaskQAModel on the combined training set.
    """
    print("Initializing Training...")
    set_seed(config.seed)

    # Load Data
    # load_cached_data=True is default, but explicit for clarity
    train_dataset = get_train_data(config, load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = MultiTaskQAModel(config)
    model.to(config.device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    num_train_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_train_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Loss Functions
    span_loss_fct = nn.CrossEntropyLoss()
    relevance_loss_fct = nn.BCEWithLogitsLoss()

    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    print(
        f"Starting training for {config.epochs} epochs on {len(train_dataset)} samples."
    )

    model.train()

    for epoch in range(config.epochs):
        total_loss = 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False
        )

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            start_positions = batch["start_positions"].to(config.device)
            end_positions = batch["end_positions"].to(config.device)
            relevance_labels = batch["relevance_labels"].to(config.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                start_logits, end_logits, relevance_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                # Compute Losses
                start_loss = span_loss_fct(start_logits, start_positions)
                end_loss = span_loss_fct(end_logits, end_positions)
                span_loss = (start_loss + end_loss) / 2

                rel_loss = relevance_loss_fct(relevance_logits, relevance_labels)

                loss = span_loss + (config.relevance_loss_weight * rel_loss)

            # Backprop
            scaler.scale(loss).backward()

            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")

    # Save Model
    print(f"Saving model to {config.best_model_path}")
    torch.save(model.state_dict(), config.best_model_path)

    # Clear memory
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()


def inference_fn(config):
    """
    Runs inference on the test set and generates the submission file.
    """
    print("Starting Inference...")
    set_seed(config.seed)

    # Load Test Data
    test_dataset = get_test_data(config, load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Load Model
    model = MultiTaskQAModel(config)
    model.load_state_dict(
        torch.load(config.best_model_path, map_location=config.device)
    )
    model.to(config.device)
    model.eval()

    # Load Raw Test CSV for Context Text Lookup
    test_df = pd.read_csv(config.test_meta_path)
    # Create mapping: id -> context
    id_to_context = dict(zip(test_df["id"], test_df["context"]))

    # Store predictions
    # Key: example_id, Value: List of (score, text) tuples
    all_predictions = collections.defaultdict(list)

    print(f"Processing {len(test_dataset)} windows...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            # Metadata
            offset_mapping = batch["offset_mapping"]  # Shape: (batch, seq_len, 2)
            example_ids = batch["example_id"]  # List of strings

            # Forward pass
            start_logits, end_logits, relevance_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            # Move to CPU
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()
            relevance_logits = relevance_logits.cpu().numpy()
            offset_mapping = offset_mapping.numpy()
            input_ids_cpu = input_ids.cpu().numpy()

            # Process batch
            for i, example_id in enumerate(example_ids):
                # Identify Context Tokens
                # XLM-R structure: <s> Q </s> </s> C </s>
                # Sep token id = 2
                sep_indices = np.where(input_ids_cpu[i] == 2)[0]

                # Context tokens are between the second </s> and the third </s>
                if len(sep_indices) >= 2:
                    context_start_idx = sep_indices[1] + 1
                    if len(sep_indices) >= 3:
                        context_end_idx = sep_indices[2] - 1
                    else:
                        context_end_idx = len(input_ids_cpu[i]) - 2
                else:
                    context_start_idx = 0
                    context_end_idx = len(input_ids_cpu[i]) - 1

                # Get logits for this sample
                s_logits = start_logits[i]
                e_logits = end_logits[i]
                rel_score = relevance_logits[i]
                offsets = offset_mapping[i]

                # Find best span in this window
                best_window_score = -float("inf")
                best_window_text = ""

                # Get top N start and end args
                start_indexes = np.argsort(s_logits)[-config.n_best_size :]
                end_indexes = np.argsort(e_logits)[-config.n_best_size :]

                for start_index in start_indexes:
                    if start_index < context_start_idx or start_index > context_end_idx:
                        continue

                    for end_index in end_indexes:
                        if end_index < context_start_idx or end_index > context_end_idx:
                            continue

                        if end_index < start_index:
                            continue

                        length = end_index - start_index + 1
                        if length > config.max_answer_length:
                            continue

                        # Calculate Score: (Start + End) + Relevance
                        span_score = (
                            s_logits[start_index] + e_logits[end_index] + rel_score
                        )

                        if span_score > best_window_score:
                            best_window_score = span_score

                            # Extract Text
                            start_char = offsets[start_index][0]
                            end_char = offsets[end_index][1]

                            context_text = id_to_context.get(example_id, "")
                            if context_text and end_char <= len(context_text):
                                best_window_text = context_text[start_char:end_char]
                            else:
                                best_window_text = ""

                if best_window_text:
                    all_predictions[example_id].append(
                        (best_window_score, best_window_text)
                    )

    # Aggregate Predictions
    final_predictions = []

    # Ensure we have a prediction for every ID in test set
    all_test_ids = test_df["id"].unique()

    for example_id in all_test_ids:
        preds = all_predictions.get(example_id, [])

        if not preds:
            final_predictions.append({"id": example_id, "PredictionString": '""'})
            continue

        # Sort by score descending
        preds.sort(key=lambda x: x[0], reverse=True)
        best_pred_text = preds[0][1]

        # Add quotes as per format requirement
        formatted_pred = f'"{best_pred_text}"'

        final_predictions.append({"id": example_id, "PredictionString": formatted_pred})

    # Save Submission
    submission_df = pd.DataFrame(final_predictions)

    # Use quoting=csv.QUOTE_NONE because we manually added quotes
    submission_df.to_csv(
        config.submission_path, index=False, quoting=csv.QUOTE_NONE, escapechar="\\"
    )
    print(f"Submission saved to {config.submission_path}")


def main():
    config = Config()

    # 1. Train
    train_fn(config)

    # 2. Inference
    inference_fn(config)
