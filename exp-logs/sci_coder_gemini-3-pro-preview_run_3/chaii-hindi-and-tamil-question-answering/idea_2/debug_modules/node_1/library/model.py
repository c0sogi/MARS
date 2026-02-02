import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter, defaultdict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders


def get_model():
    """
    Initializes the XLM-RoBERTa model for Token Classification.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        Config.MODEL_CHECKPOINT,
        num_labels=Config.NUM_LABELS,
        id2label=Config.ID2LABEL,
        label2id=Config.LABEL2ID,
    )
    model.to(Config.DEVICE)
    return model


def extract_answer(input_ids, predictions, tokenizer):
    """
    Extracts the answer string from token IDs and predicted labels.
    Strategy: Finds the first B-ANS span in the Context segment.
    """
    # Identify special tokens
    sep_token_id = tokenizer.sep_token_id

    # Locate the start of the context.
    # XLM-R format: <s> Question </s> </s> Context </s>
    # We look for the second </s> (sep_token_id) to start the context.
    # If not found (rare/truncated), we default to searching the whole sequence or after the first.
    sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

    context_start_idx = 0
    if len(sep_indices) >= 2:
        context_start_idx = sep_indices[1] + 1
    elif len(sep_indices) == 1:
        context_start_idx = sep_indices[0] + 1

    b_ans_id = Config.LABEL2ID["B-ANS"]
    i_ans_id = Config.LABEL2ID["I-ANS"]

    start_idx = -1
    # Search for B-ANS only within the context
    for i in range(context_start_idx, len(predictions)):
        if predictions[i] == b_ans_id:
            start_idx = i
            break

    if start_idx == -1:
        return ""

    # Find the end of the span (continuous I-ANS)
    end_idx = start_idx
    for i in range(start_idx + 1, len(predictions)):
        if predictions[i] == i_ans_id:
            end_idx = i
        else:
            break

    # Decode the token span
    token_ids = input_ids[start_idx : end_idx + 1]
    answer = tokenizer.decode(token_ids, skip_special_tokens=True)
    return answer.strip()


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_loss(model, dataloader, device):
    """
    Computes validation loss.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)


def predict(model, dataloader, tokenizer, device):
    """
    Runs inference and returns a dictionary {example_id: prediction_string}.
    Implements Greedy First-Match Decoding per example.
    """
    model.eval()

    # Group windows by example_id
    # We store (input_ids, predictions)
    grouped_windows = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            example_ids = batch["example_id"]  # List of strings

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            input_ids_cpu = input_ids.cpu().numpy()

            for i, ex_id in enumerate(example_ids):
                grouped_windows[ex_id].append((input_ids_cpu[i], preds[i]))

    # Decode
    results = {}
    for ex_id, windows in grouped_windows.items():
        # Windows are processed in order of the dataloader.
        # Assuming dataloader (no shuffle) preserves window order (stride).

        prediction = ""
        for inp, pred in windows:
            ans = extract_answer(inp, pred, tokenizer)
            if ans:
                prediction = ans
                break  # Stop at first valid span (Greedy First-Match)

        results[ex_id] = prediction

    return results


def run_training():
    """
    Orchestrates the training of 3 models with different seeds.
    """
    # Load Data
    train_loader, val_loader, _ = get_dataloaders(debug=False, load_cached_data=True)

    # Config
    epochs = Config.EPOCHS
    device = Config.DEVICE

    for seed in Config.SEEDS:
        print(f"\n=== Training Seed {seed} ===")
        set_seed(seed)

        model = get_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        num_training_steps = epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        save_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pt")

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device
            )
            val_loss = validate_loss(model, val_loader, device)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Clean up to save memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()


def generate_submission():
    """
    Loads models, predicts on test set, performs majority voting, saves submission.
    """
    print("\n=== Generating Submission ===")
    _, _, test_loader = get_dataloaders(debug=False, load_cached_data=True)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)
    device = Config.DEVICE

    all_predictions = []  # List of dicts

    for seed in Config.SEEDS:
        model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pt")
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} not found. Skipping.")
            continue

        print(f"Loading model from {model_path}...")
        model = get_model()
        model.load_state_dict(torch.load(model_path, map_location=device))

        preds = predict(model, test_loader, tokenizer, device)
        all_predictions.append(preds)

        del model
        torch.cuda.empty_cache()

    if not all_predictions:
        print("No predictions generated.")
        return

    # Majority Voting
    final_preds = {}
    # Get all example IDs from the first prediction dict
    # Assuming all models predicted on the same set of IDs
    if len(all_predictions) > 0:
        example_ids = list(all_predictions[0].keys())

        for ex_id in example_ids:
            votes = []
            for pred_dict in all_predictions:
                votes.append(pred_dict.get(ex_id, ""))

            # Tie-breaking: Counter.most_common returns arbitrary for ties (usually first encountered).
            # With 3 seeds, ties are rare unless 1-1-1.
            most_common = Counter(votes).most_common(1)[0][0]
            final_preds[ex_id] = most_common

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"id": list(final_preds.keys()), "PredictionString": list(final_preds.values())}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run():
    """
    Main entry point to execute the full pipeline.
    """
    run_training()
    generate_submission()
