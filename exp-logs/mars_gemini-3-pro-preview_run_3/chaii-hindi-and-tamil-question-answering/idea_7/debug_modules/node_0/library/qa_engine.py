import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel

from library.config import Config
from library.utils import set_seed, jaccard
from library.data import QADataset, qa_collate_fn
from library.model import WeightedTokenClassifier, get_class_weights


def get_predictions(model, dataloader, device):
    """
    Runs inference on a dataloader and aggregates predictions by example_id.
    Selects the span with the highest confidence score across all windows for a document.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader containing the dataset.
        device: The torch device.

    Returns:
        dict: A dictionary mapping example_id (str) to predicted answer text (str).
    """
    model.eval()

    # Dictionary to store the best candidate for each example_id
    # Structure: {example_id: {'score': float, 'text': str}}
    candidates = {}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)  # (Batch, Seq_Len, Num_Labels)

            # Iterate over each sample in the batch
            metadata = batch["metadata"]
            for i in range(len(metadata)):
                example_id = metadata[i]["example_id"]
                offsets = metadata[i]["offset_mapping"]
                context = metadata[i]["context"]
                sequence_ids = metadata[i]["sequence_ids"]

                # Get probabilities and predicted classes for this sample
                sample_probs = probs[i]  # (Seq_Len, 3)
                pred_classes = torch.argmax(sample_probs, dim=-1).cpu().numpy()

                # Identify answer spans
                # We look for continuous segments of B-ANS (1) or I-ANS (2)
                # We filter by sequence_ids to ensure we only look at the context (usually id 1)

                valid_indices = [
                    idx for idx, s_id in enumerate(sequence_ids) if s_id == 1
                ]

                current_span_indices = []
                current_span_probs = []
                spans = []

                for idx in valid_indices:
                    cls = pred_classes[idx]
                    p = sample_probs[idx, cls].item()

                    if cls != 0:  # If B-ANS or I-ANS
                        current_span_indices.append(idx)
                        current_span_probs.append(p)
                    else:
                        if current_span_indices:
                            # End of a span
                            avg_score = np.mean(current_span_probs)
                            spans.append((current_span_indices, avg_score))
                            current_span_indices = []
                            current_span_probs = []

                # Capture span if it ends at the boundary
                if current_span_indices:
                    avg_score = np.mean(current_span_probs)
                    spans.append((current_span_indices, avg_score))

                # If no spans found in this window, skip
                if not spans:
                    continue

                # Select the best span in this window based on confidence score
                best_span_indices, best_span_score = max(spans, key=lambda x: x[1])

                # Decode text using offset mapping
                try:
                    start_char = offsets[best_span_indices[0]][0]
                    end_char = offsets[best_span_indices[-1]][1]
                    pred_text = context[start_char:end_char]
                except IndexError:
                    continue

                # Global Aggregation: Update if this window has a higher confidence candidate
                if (
                    example_id not in candidates
                    or best_span_score > candidates[example_id]["score"]
                ):
                    candidates[example_id] = {
                        "score": best_span_score,
                        "text": pred_text,
                    }

    # Format return dict
    final_predictions = {eid: data["text"] for eid, data in candidates.items()}
    return final_predictions


def validate(model, dataloader, device, val_df):
    """
    Evaluates the model on the validation set using the Jaccard metric.

    Args:
        model: The trained model.
        dataloader: Validation dataloader.
        device: Torch device.
        val_df: DataFrame containing ground truth (must have 'id' and 'answer_text').

    Returns:
        float: The mean Jaccard score.
    """
    print("Starting validation...")
    predictions = get_predictions(model, dataloader, device)

    # Calculate Jaccard Score
    scores = []
    # Map ID to Ground Truth
    gt_map = dict(zip(val_df["id"], val_df["answer_text"]))

    for eid, gt_text in gt_map.items():
        # Default to empty string if no prediction made
        pred_text = predictions.get(eid, "")
        score = jaccard(gt_text, pred_text)
        scores.append(score)

    mean_jaccard = np.mean(scores)
    print(f"Validation Jaccard Score: {mean_jaccard}")
    return mean_jaccard


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.6f}")
    return avg_loss


def train_model(seed):
    """
    Full training pipeline for a single seed.
    """
    set_seed(seed)
    print(f"\n{'='*30}\nTraining Model with Seed {seed}\n{'='*30}")

    # 1. Load Data
    train_dataset = QADataset(mode="train")
    val_dataset = QADataset(mode="val")

    # Load raw validation dataframe for GT evaluation
    val_df = pd.read_csv(Config.VAL_CSV)

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

    # 2. Calculate Class Weights
    class_weights = get_class_weights(train_dataset)
    class_weights = class_weights.to(Config.DEVICE)

    # 3. Initialize Model
    model = WeightedTokenClassifier(class_weights=class_weights)

    # 4. Load TAPT Weights (if available)
    # TAPT saves a full model. We load the backbone into model.roberta
    if os.path.exists(Config.TAPT_OUTPUT_DIR):
        print(f"Loading TAPT-finetuned backbone from {Config.TAPT_OUTPUT_DIR}...")
        try:
            # Load the TAPT model (which is likely ForMaskedLM)
            # AutoModel.from_pretrained will load the base architecture (embeddings + encoder)
            # and ignore the LM head, which is exactly what we want for the backbone.
            tapt_backbone = AutoModel.from_pretrained(Config.TAPT_OUTPUT_DIR)

            # Transfer weights
            model.roberta.load_state_dict(tapt_backbone.state_dict())
            print("TAPT weights loaded successfully.")
            del tapt_backbone
        except Exception as e:
            print(f"Warning: Failed to load TAPT weights: {e}")
            print("Proceeding with base XLM-R weights.")
    else:
        print("No TAPT weights found. Using base XLM-R weights.")

    model.to(Config.DEVICE)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 6. Training Loop
    best_score = -1.0
    patience_counter = 0
    save_path = os.path.join(Config.MODEL_OUTPUT_DIR, f"model_seed_{seed}.pt")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE, epoch
        )
        val_score = validate(model, val_loader, Config.DEVICE, val_df)

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} (Score: {best_score:.6f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Clean up
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    return best_score


def generate_submission():
    """
    Generates the final submission file by ensembling predictions from all trained seeds.
    Uses Majority Voting.
    """
    print(f"\n{'='*30}\nGenerating Submission\n{'='*30}")

    # 1. Load Test Data
    test_dataset = QADataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=qa_collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    # Get list of all test IDs
    # We load the raw test CSV to ensure we have all IDs and correct order
    df_test = pd.read_csv(Config.TEST_CSV)
    test_ids = df_test["id"].tolist()

    # 2. Collect Predictions from all Seeds
    all_seed_predictions = []  # List of dicts

    for seed in Config.SEEDS:
        model_path = os.path.join(Config.MODEL_OUTPUT_DIR, f"model_seed_{seed}.pt")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        print(f"Running inference with model seed {seed}...")

        # Initialize model
        # Note: We don't need class weights for inference, but init requires arg
        model = WeightedTokenClassifier(class_weights=None)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)

        preds = get_predictions(model, test_loader, Config.DEVICE)
        all_seed_predictions.append(preds)

        del model
        torch.cuda.empty_cache()

    if not all_seed_predictions:
        print("Error: No models found for inference.")
        return

    # 3. Ensemble (Majority Vote)
    final_predictions = []

    print("Aggregating predictions...")
    for eid in test_ids:
        votes = []
        for pred_dict in all_seed_predictions:
            # Get prediction for this ID, default to empty string
            votes.append(pred_dict.get(eid, ""))

        # Majority Vote
        counter = Counter(votes)
        # most_common returns list of (element, count). We take the element of the first one.
        best_pred = counter.most_common(1)[0][0]

        final_predictions.append({"id": eid, "PredictionString": best_pred})

    # 4. Save Submission
    submission_df = pd.DataFrame(final_predictions)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_training():
    """
    Orchestrates the full training process for all seeds and generates submission.
    """
    for seed in Config.SEEDS:
        train_model(seed)

    generate_submission()
