import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AdamW
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from library.config import Config
from library.utils import set_seed, postprocess_qa_predictions
from library.data import load_data, QADataset, get_tokenizer


class QAModel(nn.Module):
    """
    Question Answering Model based on DistilBERT.
    """

    def __init__(self, model_checkpoint):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_checkpoint)
        self.dropout = nn.Dropout(0.1)
        # DistilBERT base hidden size is 768
        self.qa_outputs = nn.Linear(768, 2)

    def forward(
        self, input_ids, attention_mask, start_positions=None, end_positions=None
    ):
        """
        Forward pass of the model.

        Args:
            input_ids: Tensor of token ids.
            attention_mask: Tensor of attention masks.
            start_positions: Tensor of ground truth start indices (optional).
            end_positions: Tensor of ground truth end indices (optional).

        Returns:
            Tuple: (loss, start_logits, end_logits) if labels provided, else (start_logits, end_logits).
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # (Batch, Seq_Len, 2)
        logits = self.qa_outputs(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        if loss is not None:
            return loss, start_logits, end_logits
        else:
            return start_logits, end_logits


def train_epoch(model, dataloader, optimizer, device, scaler):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.amp.autocast("cuda"):
            loss, _, _ = model(
                input_ids, attention_mask, start_positions, end_positions
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """
    Runs validation loop.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            loss, _, _ = model(
                input_ids, attention_mask, start_positions, end_positions
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def run_training(debug=False):
    """
    Main training function.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing training (Debug={debug})...")

    # Load Data
    tokenizer = get_tokenizer()
    train_df = load_data("train", tokenizer, debug=debug)
    val_df = load_data("val", tokenizer, debug=debug)

    train_dataset = QADataset(train_df, mode="train")
    # We use mode='train' for validation dataset here to ensure it returns labels for loss calculation
    val_dataset = QADataset(val_df, mode="train")

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

    # Optimizer & Scaler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0
    model_save_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")


def generate_submission(debug=False):
    """
    Generates predictions for the test set and saves the submission file.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    tokenizer = get_tokenizer()

    print(f"Generating submission (Debug={debug})...")

    # Load Test Data
    # 1. Processed features for model input
    test_features = load_data("test", tokenizer, debug=debug)
    test_dataset = QADataset(test_features, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Raw examples for post-processing
    test_examples_df = pd.read_csv(Config.TEST_DATA_PATH)
    if debug:
        test_examples_df = test_examples_df.head(Config.DEBUG_SIZE)

    # Load Model
    model = QAModel(Config.MODEL_CHECKPOINT)
    model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(
            "Warning: Best model not found. Using initialized weights (random prediction)."
        )

    model.to(device)
    model.eval()

    all_start_logits = []
    all_end_logits = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # Post-process
    print("Post-processing predictions...")

    # Convert Pandas DataFrames to HuggingFace Datasets for compatibility with
    # the provided utils.postprocess_qa_predictions which iterates over examples/features
    hf_examples = Dataset.from_pandas(test_examples_df)
    hf_features = Dataset.from_pandas(test_features)

    predictions = postprocess_qa_predictions(
        examples=hf_examples,
        features=hf_features,
        predictions=(all_start_logits, all_end_logits),
        n_best_size=Config.N_BEST_SIZE,
        max_answer_length=Config.MAX_ANSWER_LENGTH,
    )

    # Create Submission CSV
    submission_ids = []
    submission_preds = []

    # Ensure we output rows for all IDs in the test set
    for ex_id in test_examples_df["id"]:
        pred_text = predictions.get(ex_id, "")

        # Format: quoted string as per submission requirement
        # Escape existing quotes if any (though unlikely in simple answers)
        pred_text = pred_text.replace('"', '""')
        formatted_pred = f'"{pred_text}"'

        submission_ids.append(ex_id)
        submission_preds.append(formatted_pred)

    submission_df = pd.DataFrame(
        {"id": submission_ids, "PredictionString": submission_preds}
    )

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
