import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from library.utils import seed_everything, get_device
from library.data_processing import DataProcessor

# --- Dataset ---


class TabularDataset(Dataset):
    def __init__(self, X, y=None, cat_cols=None, cont_cols=None):
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

        # Extract categorical data
        if cat_cols:
            self.x_cat = torch.tensor(X[cat_cols].values, dtype=torch.long)
        else:
            self.x_cat = None

        # Extract continuous data
        if cont_cols:
            self.x_cont = torch.tensor(X[cont_cols].values, dtype=torch.float32)
        else:
            self.x_cont = None

    def __len__(self):
        return len(self.x_cat) if self.x_cat is not None else len(self.x_cont)

    def __getitem__(self, idx):
        x_cat = self.x_cat[idx] if self.x_cat is not None else torch.empty(0)
        x_cont = self.x_cont[idx] if self.x_cont is not None else torch.empty(0)

        if self.y is not None:
            return x_cat, x_cont, self.y[idx]
        else:
            return x_cat, x_cont


# --- Model ---


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class TabularResNet(nn.Module):
    def __init__(
        self,
        cat_dims,
        num_cont,
        num_classes,
        embed_dim=16,
        hidden_dim=256,
        num_blocks=2,
        dropout_rate=0.1,
    ):
        super(TabularResNet, self).__init__()

        # Embeddings for categorical features
        # cat_dims is a list of tuples: (vocab_size, embedding_dim)
        # Here we simplify and use fixed embed_dim for all, or we could vary it.
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
                for vocab_size in cat_dims
            ]
        )

        total_embed_dim = len(cat_dims) * embed_dim

        # Input projection
        input_dim = total_embed_dim + num_cont
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        # Residual Blocks
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        # Output Head
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_cat, x_cont):
        # x_cat: [batch_size, num_cat_features]
        # x_cont: [batch_size, num_cont_features]

        # Process embeddings
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            embedded.append(emb_layer(x_cat[:, i]))

        if embedded:
            x_emb = torch.cat(embedded, dim=1)
        else:
            x_emb = torch.tensor([], device=x_cont.device)

        # Concatenate with continuous features
        x = torch.cat([x_emb, x_cont], dim=1)

        # Project and pass through blocks
        x = self.input_proj(x)
        x = self.blocks(x)

        # Output
        logits = self.head(x)
        return logits


# --- Training Helper ---


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    running_loss = 0.0

    for x_cat, x_cont, targets in loader:
        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(x_cat, x_cont)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * targets.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for x_cat, x_cont, targets in loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            targets = targets.to(device)

            logits = model(x_cat, x_cont)
            loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)

            probs = torch.softmax(logits, dim=1)
            preds_list.append(probs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)

    # Calculate accuracy
    pred_labels = np.argmax(all_preds, axis=1)
    acc = accuracy_score(all_targets, pred_labels)

    return avg_loss, acc, all_preds


def predict(model, loader, device):
    model.eval()
    preds_list = []

    with torch.no_grad():
        for x_cat, x_cont in loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)

            logits = model(x_cat, x_cont)
            probs = torch.softmax(logits, dim=1)
            preds_list.append(probs.cpu().numpy())

    return np.concatenate(preds_list, axis=0)


# --- Main CV Function ---


def run_resnet_cv(
    load_cached_data=True, n_splits=5, seed=42, batch_size=2048, epochs=50, patience=10
):
    """
    Executes Stratified 5-Fold CV for the ResNet backbone.
    """
    seed_everything(seed)
    device = get_device()
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Initializing DataProcessor for ResNet...")
    processor = DataProcessor()
    X_train_part, y_train_part, X_val_part, y_val_part, X_test, le = (
        processor.get_nn_data(load_cached_data=load_cached_data)
    )

    # Combine for full CV
    print("Combining initial Train/Val splits for full Stratified 5-Fold CV...")
    X_full = pd.concat([X_train_part, X_val_part], axis=0).reset_index(drop=True)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)

    print(f"Full Training Shape: {X_full.shape}")
    print(f"Test Shape: {X_test.shape}")

    # Identify Feature Columns
    cat_cols = ["Soil_Type_Index", "Wilderness_Area_Index"]
    cont_cols = [c for c in X_full.columns if c not in cat_cols]

    # Calculate vocab sizes for embeddings
    # We add 1 because indices are 1-based (or at least >0), and we want to be safe.
    # Max index found + 1 is the vocab size.
    cat_dims = []
    for col in cat_cols:
        # Ensure we cover the max index in both train and test
        max_idx = max(X_full[col].max(), X_test[col].max())
        cat_dims.append(int(max_idx) + 1)

    print(f"Categorical Features: {cat_cols}")
    print(f"Vocab Sizes: {cat_dims}")
    print(f"Continuous Features: {len(cont_cols)}")

    # --- Setup ---
    num_classes = len(le.classes_)
    oof_preds = np.zeros((len(X_full), num_classes), dtype=np.float32)
    test_preds_sum = np.zeros((len(X_test), num_classes), dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Prepare Test Loader (once)
    test_dataset = TabularDataset(X_test, cat_cols=cat_cols, cont_cols=cont_cols)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    fold_scores = []

    # --- CV Loop ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Split Data
        X_tr, y_tr = X_full.iloc[train_idx], y_full[train_idx]
        X_va, y_va = X_full.iloc[val_idx], y_full[val_idx]

        # Create Datasets and Loaders
        train_dataset = TabularDataset(
            X_tr, y_tr, cat_cols=cat_cols, cont_cols=cont_cols
        )
        val_dataset = TabularDataset(X_va, y_va, cat_cols=cat_cols, cont_cols=cont_cols)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Initialize Model
        model = TabularResNet(
            cat_dims=cat_dims,
            num_cont=len(cont_cols),
            num_classes=num_classes,
            embed_dim=16,
            hidden_dim=256,
            num_blocks=2,
            dropout_rate=0.1,
        ).to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # OneCycleLR
        total_steps = epochs * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=total_steps, pct_start=0.1
        )

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scheduler
            )
            val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)

            # Print metrics
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
            )

            # Early Stopping Check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final Validation Inference
        _, final_acc, val_probs = evaluate(model, val_loader, criterion, device)
        oof_preds[val_idx] = val_probs
        fold_scores.append(final_acc)

        print(f"Fold {fold+1} Best Accuracy: {final_acc}")

        # Test Inference
        test_probs = predict(model, test_loader, device)
        test_preds_sum += test_probs

        # Cleanup
        del (
            model,
            optimizer,
            scheduler,
            train_dataset,
            val_dataset,
            train_loader,
            val_loader,
        )
        torch.cuda.empty_cache()

    # --- Aggregation ---
    avg_test_preds = test_preds_sum / n_splits
    mean_acc = np.mean(fold_scores)

    print(f"\nResNet CV Complete.")
    print(f"Average Accuracy: {mean_acc}")

    return oof_preds, avg_test_preds, le, y_full
