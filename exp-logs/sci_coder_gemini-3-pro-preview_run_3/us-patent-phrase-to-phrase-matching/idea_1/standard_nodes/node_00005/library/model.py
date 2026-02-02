import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
from transformers import AutoModel

# Import from provided library files
from library.config import Config, set_seed
from library.data_loader import get_dataloaders


class PatentBert(nn.Module):
    """
    Transformer-based model for phrase similarity.
    Cite solution_lesson_node_00001: Replaces DAN with pre-trained Transformer.
    """

    def __init__(self, model_name):
        super(PatentBert, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token embedding (index 0)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        score = self.fc(cls_emb)
        return score.squeeze(-1)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["score"].to(device)

        optimizer.zero_grad()

        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["score"].to(device)

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)

            running_loss += loss.item() * input_ids.size(0)

            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate Pearson Correlation
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    if torch.std(all_preds) == 0 or torch.std(all_labels) == 0:
        pearson = 0.0
    else:
        vx = all_preds - torch.mean(all_preds)
        vy = all_labels - torch.mean(all_labels)
        pearson = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
        )
        pearson = pearson.item()

    return epoch_loss, pearson


def generate_submission(model, test_loader, device, output_path):
    model.eval()
    ids = []
    scores = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_ids = batch["id"]

            predictions = model(input_ids, attention_mask)

            # Clip predictions to valid range [0, 1]
            predictions = torch.clamp(predictions, 0.0, 1.0)

            ids.extend(batch_ids)
            scores.extend(predictions.cpu().numpy())

    df_sub = pd.DataFrame({"id": ids, "score": scores})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_task():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = PatentBert(Config.MODEL_NAME).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training Loop with Early Stopping
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pearson = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Pearson: {val_pearson:.6f}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 5. Inference
    print("Loading best model for inference...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Model file not found, using current model weights.")

    print("Generating submission...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


# Execute the task
run_task()
