import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import utilities from the provided library
from library.config import Config, TabularDataset, process_data, set_seed


class InputInjectedFunnelMLP(nn.Module):
    def __init__(self, cont_dim, vocab_sizes, embed_dim, hidden_dims, dropout):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # Input dimension = continuous + (num_categorical * embed_dim)
        self.input_dim = cont_dim + (len(vocab_sizes) * embed_dim)

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout)
        )

        # Layer 2 (Input Injection: Concat Previous Layer + Original Input)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0] + self.input_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Layer 3 (Input Injection)
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dims[1] + self.input_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output Head
        self.head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x_cont, x_cat):
        # Embeddings
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)

        # Concatenate continuous and embeddings -> x_in
        x_in = torch.cat([x_cont, x_emb], dim=1)

        # Forward Pass with Input Injection
        h1 = self.layer1(x_in)

        h1_inj = torch.cat([h1, x_in], dim=1)
        h2 = self.layer2(h1_inj)

        h2_inj = torch.cat([h2, x_in], dim=1)
        h3 = self.layer3(h2_inj)

        out = self.head(h3)
        return out


def train_and_predict(epochs=Config.EPOCHS, max_samples=None):
    """
    Trains the InputInjectedFunnelMLP model and generates predictions.

    Args:
        epochs (int): Number of training epochs.
        max_samples (int, optional): If set, limits the number of training samples for debugging.
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process Data (uses caching from library)
    data = process_data(load_cached_data=True)

    train_cont = data["train_cont"]
    train_cat = data["train_cat"]
    train_y = data["train_y"]

    # Apply debugging subset if requested
    if max_samples is not None and max_samples < len(train_cont):
        print(f"Debugging: Subsetting training data to {max_samples} samples.")
        train_cont = train_cont[:max_samples]
        train_cat = train_cat[:max_samples]
        train_y = train_y[:max_samples]

    # Create Datasets
    train_ds = TabularDataset(train_cont, train_cat, train_y)
    val_ds = TabularDataset(data["val_cont"], data["val_cat"], data["val_y"])
    test_ds = TabularDataset(data["test_cont"], data["test_cat"])

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Init Model
    model = InputInjectedFunnelMLP(
        cont_dim=data["cont_dim"],
        vocab_sizes=data["vocab_sizes"],
        embed_dim=Config.EMBEDDING_DIM,
        hidden_dims=Config.HIDDEN_DIMS,
        dropout=Config.DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = 0
    patience_counter = 0
    best_model_state = None

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_cont, x_cat, y in train_loader:
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_cont, x_cat)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_cont, x_cat, y in val_loader:
                x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)
                logits = model(x_cont, x_cat)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val AUC: {val_auc}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best model
            torch.save(best_model_state, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Validation AUC: {best_auc}")

    # Inference
    print("Generating predictions...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    elif os.path.exists(best_model_path):
        print("Loading best model from disk...")
        model.load_state_dict(torch.load(best_model_path))

    model.eval()

    test_preds = []
    with torch.no_grad():
        for x_cont, x_cat in test_loader:
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)
            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()

    # Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df = pd.DataFrame({"id": data["test_ids"], "target": test_preds})
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
