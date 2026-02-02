import os
import torch
import numpy as np
import collections
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from library.configuration import Config
from library.utils import seed_everything, load_data, compute_average_jaccard
from library.dataset_factory import prepare_qa_data, qa_collate_fn, get_tokenizer
from library.qa_model import XLMRobertaForQA


def extract_answer_spans(input_ids, logits, sequence_ids, tokenizer):
    """
    Extracts candidate answer spans from a single window based on BIO tagging.
    Returns a list of (score, text) tuples.
    """
    # Softmax for probabilities
    probs = torch.softmax(logits, dim=-1)  # (Seq_Len, 3)
    preds = torch.argmax(logits, dim=-1)  # (Seq_Len)

    probs = probs.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    input_ids = input_ids.detach().cpu().numpy()

    candidates = []
    seq_len = len(input_ids)
    i = 0

    while i < seq_len:
        # Skip non-context tokens (assuming context has sequence_id == 1)
        # Note: sequence_ids can contain None for special tokens
        if sequence_ids[i] != 1:
            i += 1
            continue

        # Check for Beginning of Answer (B-ANS = 1)
        if preds[i] == 1:
            start = i
            end = i

            # Look ahead for Inside of Answer (I-ANS = 2)
            # Must also be part of the context
            while (
                (end + 1) < seq_len
                and preds[end + 1] == 2
                and sequence_ids[end + 1] == 1
            ):
                end += 1

            # Calculate score: Mean probability of the predicted classes in the span
            # B-ANS probability at start, I-ANS probabilities for the rest
            span_probs = [probs[start][1]]
            for k in range(start + 1, end + 1):
                span_probs.append(probs[k][2])

            score = np.mean(span_probs)

            # Decode text directly from token IDs
            span_ids = input_ids[start : end + 1]
            text = tokenizer.decode(span_ids, skip_special_tokens=True).strip()

            candidates.append((score, text))

            i = end + 1
        else:
            i += 1

    return candidates


def evaluate_model(model, dataloader, gt_map, tokenizer, device):
    """
    Evaluates the model on the validation set using Jaccard score.
    Aggregates predictions across sliding windows for each document.
    """
    model.eval()

    # Store candidates per example_id
    # key: example_id, value: list of (score, text)
    all_candidates = collections.defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]

            # Process each sample in the batch
            batch_size = input_ids.size(0)
            for b in range(batch_size):
                eid = batch["example_id"][b]
                seq_ids = batch["sequence_ids"][b]

                # Extract spans
                spans = extract_answer_spans(
                    input_ids[b], logits[b], seq_ids, tokenizer
                )

                all_candidates[eid].extend(spans)

    # Aggregate and Compute Jaccard
    pred_strings = []
    gt_strings = []

    # Iterate over the ground truth map to ensure we cover all validation examples
    for eid, true_answer in gt_map.items():
        candidates = all_candidates.get(eid, [])

        if not candidates:
            # No valid span found (model predicted O everywhere)
            pred_text = ""
        else:
            # Select best candidate by score
            best_candidate = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
            pred_text = best_candidate[1]

        gt_strings.append(true_answer)
        pred_strings.append(pred_text)

    avg_jaccard = compute_average_jaccard(gt_strings, pred_strings)
    return avg_jaccard


def train_model(seed, pretrained_path=None, load_cached_data=True):
    """
    Trains the QA model using the specified seed and optional pretrained weights.

    Args:
        seed (int): Random seed for reproducibility.
        pretrained_path (str, optional): Path to the TAPT-finetuned model directory.
        load_cached_data (bool): Whether to load pre-processed features from cache.

    Returns:
        str: Path to the saved best model checkpoint.
    """
    # 1. Setup
    seed_everything(seed)
    device = Config.DEVICE

    # 2. Data Preparation
    tokenizer = get_tokenizer()

    print(f"Loading QA data for seed {seed}...")
    train_dataset = prepare_qa_data(
        tokenizer, split="train", load_cached_data=load_cached_data
    )
    val_dataset = prepare_qa_data(
        tokenizer, split="val", load_cached_data=load_cached_data
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=qa_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=qa_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Ground Truth for Validation Map (id -> answer_text)
    val_df = load_data("val")
    gt_map = {str(row["id"]): str(row["answer_text"]) for _, row in val_df.iterrows()}

    # 3. Model Initialization
    model = XLMRobertaForQA(pretrained_path=pretrained_path)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * Config.EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 5. Training Loop
    best_jaccard = -1.0
    best_model_path = os.path.join(Config.QA_MODELS_DIR, f"model_seed_{seed}.pt")

    print(f"Starting QA training for Seed {seed}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0.0

        # Training Step
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs[0]

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation Step
        val_jaccard = evaluate_model(model, val_loader, gt_map, tokenizer, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Jaccard: {val_jaccard:.16f}"
        )

        # Save Best Model
        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            print(f"New best model found! Saving to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)

    print(f"Training finished for Seed {seed}. Best Val Jaccard: {best_jaccard:.16f}")

    # Clean up to save memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_model_path
