import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from library.config import Config
from library.modeling import DEConvNet, WindowLogisticRegressor
from library.utils import compute_micro_f1


def train_ranker(
    model, train_loader, val_loader, val_metadata, epochs=Config.NUM_EPOCHS, device=None
):
    """
    Trains the Long Answer Ranking model (DEConvNet) using Binary Cross Entropy loss.
    Monitors Micro F1 scores on the validation set and implements Early Stopping.

    Args:
        model (nn.Module): The DEConvNet model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        val_metadata (pd.DataFrame): Metadata for validation set (for F1 calculation).
        epochs (int): Number of training epochs.
        device (torch.device): Device to train on.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCELoss()

    best_f1 = -1.0
    patience_counter = 0

    print("Starting Long Answer Ranker Training...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            q = batch["question"].to(device)
            c = batch["candidate"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            preds = model(q, c)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * q.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0

        # Dictionary to store the best candidate score and index for each example
        # example_id -> (best_score, best_candidate_index)
        best_candidates = {}

        with torch.no_grad():
            for batch in val_loader:
                q = batch["question"].to(device)
                c = batch["candidate"].to(device)
                labels = batch["label"].to(device)
                e_ids = batch["example_id"]
                c_idxs = batch["candidate_index"]

                preds = model(q, c)
                loss = criterion(preds, labels)
                val_loss += loss.item() * q.size(0)

                scores = preds.cpu().numpy()
                c_idxs_np = c_idxs.cpu().numpy()

                for eid, c_idx, score in zip(e_ids, c_idxs_np, scores):
                    if eid not in best_candidates:
                        best_candidates[eid] = (-1.0, -1)

                    if score > best_candidates[eid][0]:
                        best_candidates[eid] = (float(score), int(c_idx))

        val_loss /= len(val_loader.dataset)

        # --- F1 Calculation ---
        # Map best candidates to token spans for F1 evaluation
        predictions = {}

        # We need to read the source file to get the token spans for the selected candidates
        # To optimize, we iterate through the metadata once
        with open(Config.TRAIN_DATA_FILE, "rb") as f:
            for _, row in val_metadata.iterrows():
                eid = row["example_id"]

                # Default empty prediction
                predictions[eid] = {"long": "", "short": ""}

                if eid not in best_candidates:
                    continue

                score, cand_idx = best_candidates[eid]

                # Apply threshold
                if score < Config.LONG_ANSWER_THRESHOLD or cand_idx == -1:
                    continue

                # Read candidate info
                f.seek(row["byte_offset"])
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    candidates = data.get("long_answer_candidates", [])
                    if cand_idx < len(candidates):
                        cand = candidates[cand_idx]
                        long_ans_str = f"{cand['start_token']}:{cand['end_token']}"
                        predictions[eid]["long"] = long_ans_str
                except Exception:
                    pass

        metrics = compute_micro_f1(predictions, val_metadata)
        val_f1 = metrics["long_f1"]

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Long F1: {val_f1:.6f}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), Config.LONG_ANSWER_MODEL_PATH)
            print("  Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("  Early stopping triggered.")
                break

    # Load best model weights before returning
    if os.path.exists(Config.LONG_ANSWER_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.LONG_ANSWER_MODEL_PATH, map_location=device)
        )

    return model


def train_extractor(X, y, epochs=50, device=None):
    """
    Trains the Short Answer Extractor (WindowLogisticRegressor).
    Saves the learned weights to disk.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        epochs (int): Number of epochs.
        device (torch.device): Device to train on.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting Short Answer Extractor Training...")

    # Convert numpy arrays to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    model = WindowLogisticRegressor(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(X)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

    # Save model weights
    weights = model.linear.weight.detach().cpu().numpy()
    bias = model.linear.bias.detach().cpu().numpy()

    os.makedirs(os.path.dirname(Config.SHORT_ANSWER_WEIGHTS_PATH), exist_ok=True)
    np.save(Config.SHORT_ANSWER_WEIGHTS_PATH, {"weights": weights, "bias": bias})
    print(f"Short answer weights saved to {Config.SHORT_ANSWER_WEIGHTS_PATH}")

    return model
