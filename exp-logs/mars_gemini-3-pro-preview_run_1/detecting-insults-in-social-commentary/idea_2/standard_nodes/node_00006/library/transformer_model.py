import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.data_loader import load_data, InsultDataset


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When using deterministic algorithms, some operations might be slower or raise errors
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for insult detection.
    Wraps a pre-trained model (e.g., RoBERTa) with a linear classification head.
    """

    def __init__(self, model_name, dropout_rate=0.3):
        super(TransformerClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(dropout_rate)
        # The hidden size is usually 768 for base models
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # RoBERTa returns a tuple/object. We want the representation of the CLS token
        # or the pooled output.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use pooler_output if available (e.g. BERT), otherwise use CLS token embedding (e.g. RoBERTa)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Last hidden state is [batch_size, seq_len, hidden_size]
            # CLS token is at index 0
            pooled_output = outputs.last_hidden_state[:, 0, :]

        output = self.drop(pooled_output)
        return self.out(output)


def train_model(
    model, train_loader, val_loader, device, epochs, lr, weight_decay, patience
):
    """
    Trains the transformer model with Early Stopping based on Validation AUC.
    """
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters, lr=lr)

    # Scheduler
    num_train_steps = int(len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_transformer_model.bin")

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["labels"].to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(ids, mask)
            loss = criterion(outputs, targets.view(-1, 1))

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_preds, val_targets = predict(model, val_loader, device, return_targets=True)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> Validation AUC improved. Model saved.")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model
    print(f"Loading best model with Val AUC: {best_auc}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def predict(model, loader, device, return_targets=False):
    """
    Generates predictions using the model.
    """
    model.eval()
    final_outputs = []
    final_targets = []

    with torch.no_grad():
        for data in loader:
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)

            outputs = model(ids, mask)
            # Apply sigmoid to logits to get probabilities
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
            final_outputs.extend(outputs)

            if return_targets and "labels" in data:
                targets = data["labels"].cpu().detach().numpy().tolist()
                final_targets.extend(targets)

    # Flatten list of lists [[p1], [p2]] -> [p1, p2]
    flat_outputs = np.array([item for sublist in final_outputs for item in sublist])

    if return_targets:
        flat_targets = np.array(final_targets)
        return flat_outputs, flat_targets

    return flat_outputs


def run_transformer(train_df=None, val_df=None, test_df=None, load_cached_data=True):
    """
    Main entry point for the Semantic Branch.
    Loads data, trains the Transformer, and returns predictions.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    # If dataframes are not provided, load them using the data_loader
    if train_df is None or val_df is None or test_df is None:
        train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    print(f"Initializing Transformer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 2. Prepare Datasets
    # Note: We use the raw 'Comment' column, not 'clean_comment'
    train_dataset = InsultDataset(
        texts=train_df["Comment"].values,
        tokenizer=tokenizer,
        labels=train_df["Insult"].values,
        max_len=Config.MAX_LEN,
    )

    val_dataset = InsultDataset(
        texts=val_df["Comment"].values,
        tokenizer=tokenizer,
        labels=val_df["Insult"].values,
        max_len=Config.MAX_LEN,
    )

    test_dataset = InsultDataset(
        texts=test_df["Comment"].values,
        tokenizer=tokenizer,
        labels=None,
        max_len=Config.MAX_LEN,
    )

    # 3. Data Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Initialize Model
    model = TransformerClassifier(Config.MODEL_NAME)
    model.to(device)

    # 5. Train
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 6. Generate Predictions
    print("Generating predictions...")
    # Get validation predictions (re-run on best model)
    val_preds = predict(model, val_loader, device, return_targets=False)

    # Get test predictions
    test_preds = predict(model, test_loader, device, return_targets=False)

    return val_preds, test_preds
