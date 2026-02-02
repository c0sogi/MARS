import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import compute_qwk


def train_fn(model, dataloader, optimizer, scheduler, device):
    """
    Performs one epoch of training.
    """
    model.train()
    final_loss = 0
    count = 0

    # Binary Cross Entropy with Logits for Ordinal Regression
    # Compares logits against binary vectors (e.g., [1, 1, 0, 0, 0])
    criterion = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()
        if scheduler:
            scheduler.step()

        final_loss += loss.item() * input_ids.size(0)
        count += input_ids.size(0)

    return final_loss / count


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and QWK score.
    """
    model.eval()
    final_loss = 0
    count = 0

    criterion = nn.BCEWithLogitsLoss()

    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            final_loss += loss.item() * input_ids.size(0)
            count += input_ids.size(0)

            # Calculate predicted score: 1 + sum(sigmoid(logits))
            # This converts the 5 binary probabilities into a continuous score
            probs = torch.sigmoid(logits)
            pred_score = 1.0 + probs.sum(dim=1)

            # Calculate target score from ordinal labels: 1 + sum(labels)
            target_score = 1.0 + labels.sum(dim=1)

            preds.extend(pred_score.cpu().numpy())
            targets.extend(target_score.cpu().numpy())

    avg_loss = final_loss / count

    # Round predictions to nearest integer for QWK calculation
    preds_rounded = np.round(preds).astype(int)
    # Clip to valid range 1-6
    preds_rounded = np.clip(preds_rounded, 1, 6)

    targets_int = np.array(targets).astype(int)

    qwk = compute_qwk(targets_int, preds_rounded)

    return avg_loss, qwk


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path,
    patience=3,
):
    """
    Orchestrates the training process with early stopping.
    """
    best_qwk = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device)
        val_loss, val_qwk = eval_fn(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val QWK: {val_qwk}"
        )

        # Early Stopping Logic based on QWK
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), save_path)
            print(f"Validation QWK improved. Model saved to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in QWK. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training completed. Best Val QWK: {best_qwk}")

    # Reload best model for further use (e.g., inference)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    model.eval()
    ids = []
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            essay_ids = batch["essay_ids"]  # List of strings

            logits = model(input_ids, attention_mask)

            # Calculate predicted score: 1 + sum(sigmoid(logits))
            probs = torch.sigmoid(logits)
            pred_score = 1.0 + probs.sum(dim=1)

            # Round to integer for submission
            pred_score_int = torch.round(pred_score).cpu().numpy().astype(int)
            pred_score_int = np.clip(pred_score_int, 1, 6)

            ids.extend(essay_ids)
            preds.extend(pred_score_int)

    # Create DataFrame
    df_sub = pd.DataFrame({"essay_id": ids, "score": preds})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
