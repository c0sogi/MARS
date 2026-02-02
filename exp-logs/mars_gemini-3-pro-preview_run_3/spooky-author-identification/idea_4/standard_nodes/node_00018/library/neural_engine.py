import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)

from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data_manager import AuthorDataset, LABEL_MAP

# Suppress Transformers logging
hf_logging.set_verbosity_error()


class MLMDataset(Dataset):
    """
    Simple dataset wrapper for Masked Language Modeling.
    Accepts a list of texts and returns them for the DataCollator.
    """

    def __init__(self, texts, tokenizer, max_length=Config.MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Tokenize without padding here; DataCollator handles dynamic padding
        # However, for simplicity and safety with memory, we can truncate/pad to max_length
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


class NeuralClassifier(nn.Module):
    """
    Neural Classifier wrapping a Transformer backbone.
    Uses Concatenated [CLS] tokens from the last 4 layers for classification.
    """

    def __init__(self, backbone_path, num_classes=3, dropout_rate=0.1):
        super(NeuralClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_path, output_hidden_states=True
        )
        self.dropout = nn.Dropout(dropout_rate)

        # Concatenating last 4 layers
        self.classifier = nn.Linear(self.backbone.config.hidden_size * 4, num_classes)

        # Initialize weights of the head
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Get hidden states from the last 4 layers
        # outputs.hidden_states is a tuple of tensors
        hidden_states = outputs.hidden_states[-4:]

        # Extract [CLS] token (index 0) from each of the last 4 layers
        # Each tensor has shape (batch_size, seq_len, hidden_size)
        cls_tokens = [layer[:, 0, :] for layer in hidden_states]

        # Concatenate along the feature dimension
        # Shape: (batch_size, hidden_size * 4)
        concat_cls = torch.cat(cls_tokens, dim=1)

        x = self.dropout(concat_cls)
        logits = self.classifier(x)
        return logits


def run_mlm_pretraining(model_name, texts, load_cached_data=True):
    """
    Performs Domain-Adaptive Pre-training (DAPT) using Masked Language Modeling.

    Args:
        model_name (str): HuggingFace model identifier.
        texts (list): List of text strings for the corpus.
        load_cached_data (bool): Whether to load a previously saved adapted model.

    Returns:
        str: Path to the adapted model directory.
    """
    seed_everything()

    safe_name = model_name.replace("/", "-")
    output_dir = os.path.join(Config.WORKING_DIR, f"mlm_{safe_name}")

    # Check cache
    if load_cached_data and os.path.exists(output_dir):
        # Check for essential files
        if os.path.exists(os.path.join(output_dir, "config.json")) and (
            os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))
            or os.path.exists(os.path.join(output_dir, "model.safetensors"))
        ):
            print(f"Loading cached MLM-adapted model from {output_dir}")
            return output_dir

    print(f"Starting MLM pre-training for {model_name}...")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(Config.DEVICE)
    model.train()

    # Prepare Data
    dataset = MLMDataset(texts, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_PROBABILITY
    )

    dataloader = DataLoader(
        dataset,
        batch_size=Config.MLM_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scaler for AMP
    scaler = torch.amp.GradScaler("cuda")

    print(f"  Corpus size: {len(texts)} samples")
    print(f"  Epochs: {Config.MLM_EPOCHS}")

    for epoch in range(Config.MLM_EPOCHS):
        total_loss = 0
        steps = 0
        start_time = time.time()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            labels = batch["labels"].to(Config.DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0
        print(
            f"  MLM Epoch {epoch+1}/{Config.MLM_EPOCHS} | Loss: {avg_loss:.6f} | Time: {time.time() - start_time:.1f}s"
        )

    # Save adapted model
    print(f"Saving MLM-adapted model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


def train_classifier(
    model_name, backbone_path, train_texts, train_labels, val_texts, val_labels
):
    """
    Fine-tunes the adapted backbone for classification.

    Args:
        model_name (str): Original model name (for naming purposes).
        backbone_path (str): Path to the MLM-adapted model.
        train_texts, train_labels: Training data.
        val_texts, val_labels: Validation data.

    Returns:
        tuple: (best_model, tokenizer, best_val_loss)
    """
    seed_everything()
    print(f"Starting classification fine-tuning for {model_name}...")

    # Load Tokenizer from the adapted path
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)

    # Prepare Datasets
    train_dataset = AuthorDataset(train_texts, train_labels, tokenizer)
    val_dataset = AuthorDataset(val_texts, val_labels, tokenizer)

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
    model = NeuralClassifier(backbone_path, num_classes=3)
    model.to(Config.DEVICE)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * Config.WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    # Tracking
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss_sum = 0
        steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            labels = batch["labels"].to(Config.DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item()
            steps += 1

        avg_train_loss = train_loss_sum / steps if steps > 0 else 0

        # --- Validation ---
        model.eval()
        val_loss_sum = 0
        val_steps = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(Config.DEVICE)
                attention_mask = batch["attention_mask"].to(Config.DEVICE)
                labels = batch["labels"].to(Config.DEVICE)

                with torch.amp.autocast("cuda"):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                val_loss_sum += loss.item()
                val_steps += 1

                probs = torch.softmax(logits, dim=1)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0

        # Calculate Metric (Log Loss)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        metric_score = compute_metric(all_labels, all_preds)

        print(
            f"  Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Metric: {metric_score}"
        )

        # --- Early Stopping ---
        # We monitor the actual Metric (Log Loss) as per competition goal
        if metric_score < best_val_loss:
            best_val_loss = metric_score
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best checkpoint immediately
            save_path = os.path.join(
                Config.WORKING_DIR, f"best_finetuned_{model_name.replace('/', '-')}.pt"
            )
            torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, tokenizer, best_val_loss


def predict_neural(model, tokenizer, texts):
    """
    Generates probability predictions for a list of texts.

    Args:
        model (NeuralClassifier): Trained model.
        tokenizer: Corresponding tokenizer.
        texts (list): List of text strings.

    Returns:
        np.ndarray: Probability matrix of shape (n_samples, 3).
    """
    model.eval()
    model.to(Config.DEVICE)

    dataset = AuthorDataset(texts, labels=None, tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)

            with torch.amp.autocast("cuda"):
                logits = model(input_ids, attention_mask)

            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
