import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import Dataset

from library.config import Config
from library.utils import set_seed, postprocess_qa_predictions, jaccard
from library.data import load_data, QADataset, get_tokenizer
from library.model import QAModel


def train_one_epoch(model, dataloader, optimizer, device, scaler):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast("cuda"):
            loss, _, _ = model(
                input_ids, attention_mask, start_positions, end_positions
            )

        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device, examples_df, features_df):
    """
    Validates the model by computing the Jaccard score on the validation set.
    """
    model.eval()
    all_start_logits = []
    all_end_logits = []

    # Inference loop
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get logits
            start_logits, end_logits = model(input_ids, attention_mask)

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())

    # Concatenate logits
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # Prepare for post-processing
    hf_examples = Dataset.from_pandas(examples_df)
    hf_features = Dataset.from_pandas(features_df)

    # Decode predictions
    predictions = postprocess_qa_predictions(
        examples=hf_examples,
        features=hf_features,
        predictions=(all_start_logits, all_end_logits),
        n_best_size=Config.N_BEST_SIZE,
        max_answer_length=Config.MAX_ANSWER_LENGTH,
    )

    # Compute Jaccard Score
    scores = []
    # Create a lookup for ground truth
    gt_map = dict(zip(examples_df["id"], examples_df["answer_text"]))

    for ex_id, pred_text in predictions.items():
        if ex_id in gt_map:
            gt_text = gt_map[ex_id]
            scores.append(jaccard(gt_text, pred_text))
        else:
            scores.append(0.0)

    return np.mean(scores)


def run_training(debug=False):
    """
    Orchestrates the training process including data loading, model initialization,
    training loop, validation, and early stopping.
    """
    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = Config.DEVICE
    tokenizer = get_tokenizer()

    print(f"Loading data (Debug={debug})...")
    # Load processed features (cached)
    train_features = load_data("train", tokenizer, debug=debug)
    val_features = load_data("val", tokenizer, debug=debug)

    # Load raw validation examples for Jaccard computation
    val_examples = pd.read_csv(Config.VAL_DATA_PATH)
    if debug:
        val_examples = val_examples.head(Config.DEBUG_SIZE)

    # Create Datasets
    train_dataset = QADataset(train_features, mode="train")
    val_dataset = QADataset(val_features, mode="val")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = QAModel(Config.MODEL_CHECKPOINT)
    model.to(device)

    # Optimizer and Scaler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda")

    # Training Loop
    best_jaccard = -1.0
    patience = 2
    patience_counter = 0
    model_save_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_jaccard = validate(model, val_loader, device, val_examples, val_features)

        # Print full precision as requested
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Jaccard: {val_jaccard}"
        )

        # Save best model
        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with Jaccard: {best_jaccard}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Jaccard: {best_jaccard}")


def generate_submission(debug=False):
    """
    Generates predictions for the test set and saves them to the submission file.
    """
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    set_seed(Config.SEED)
    device = Config.DEVICE
    tokenizer = get_tokenizer()

    print(f"Generating submission (Debug={debug})...")

    # Load test data
    test_features = load_data("test", tokenizer, debug=debug)
    test_examples = pd.read_csv(Config.TEST_DATA_PATH)
    if debug:
        test_examples = test_examples.head(Config.DEBUG_SIZE)

    test_dataset = QADataset(test_features, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load best model
    model = QAModel(Config.MODEL_CHECKPOINT)
    model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using random initialization.")

    model.to(device)
    model.eval()

    all_start_logits = []
    all_end_logits = []

    # Inference
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # Post-processing
    hf_examples = Dataset.from_pandas(test_examples)
    hf_features = Dataset.from_pandas(test_features)

    predictions = postprocess_qa_predictions(
        examples=hf_examples,
        features=hf_features,
        predictions=(all_start_logits, all_end_logits),
        n_best_size=Config.N_BEST_SIZE,
        max_answer_length=Config.MAX_ANSWER_LENGTH,
    )

    # Format submission
    submission_ids = []
    submission_preds = []

    # Iterate over original test IDs to ensure correct order/completeness
    for ex_id in test_examples["id"]:
        pred_text = predictions.get(ex_id, "")

        # Format as quoted string
        pred_text = pred_text.replace('"', '""')
        formatted_pred = f'"{pred_text}"'

        submission_ids.append(ex_id)
        submission_preds.append(formatted_pred)

    submission_df = pd.DataFrame(
        {"id": submission_ids, "PredictionString": submission_preds}
    )

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
