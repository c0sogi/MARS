import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from library.utils import get_device, seed_everything
from library.dataset import get_datasets


class ManufacturingNet(nn.Module):
    """
    Deep Neural Network with Entity Embeddings for sequence data and
    an MLP backbone for combined features.
    """

    def __init__(
        self,
        num_numerical_features,
        vocab_size,
        embedding_dim,
        seq_len,
        hidden_units=[512, 256, 128],
        dropout_rate=0.3,
    ):
        super(ManufacturingNet, self).__init__()

        # Entity Embedding for sequence characters
        # padding_idx=0 is used for unknown/padding characters
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.seq_flatten_dim = seq_len * embedding_dim

        # Calculate input dimension for the MLP
        input_dim = num_numerical_features + self.seq_flatten_dim

        # Build MLP Backbone
        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Final Output Layer
        self.output_layer = nn.Linear(curr_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, numerical, sequence):
        """
        Args:
            numerical: Tensor of shape (batch_size, num_features)
            sequence: Tensor of shape (batch_size, seq_len)
        """
        # 1. Process Sequence
        # (batch, seq_len) -> (batch, seq_len, emb_dim)
        emb = self.embedding(sequence)
        # Flatten: (batch, seq_len * emb_dim)
        emb_flat = emb.view(emb.size(0), -1)

        # 2. Combine Features
        x = torch.cat([numerical, emb_flat], dim=1)

        # 3. MLP Pass
        x = self.mlp(x)

        # 4. Output Probability
        logits = self.output_layer(x)
        probs = self.sigmoid(logits)

        return probs


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for batch in dataloader:
        numerical = batch["numerical"].to(device)
        sequence = batch["sequence"].to(device)
        targets = batch["label"].to(device).unsqueeze(1)  # Shape: (batch, 1)

        optimizer.zero_grad()
        outputs = model(numerical, sequence)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * targets.size(0)

        # Detach and store for AUC calculation
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(outputs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            numerical = batch["numerical"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["label"].to(device).unsqueeze(1)

            outputs = model(numerical, sequence)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def generate_submission(
    model, test_loader, device, output_path="./submission/submission.csv"
):
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in test_loader:
            numerical = batch["numerical"].to(device)
            sequence = batch["sequence"].to(device)
            ids = batch["id"].numpy()

            outputs = model(numerical, sequence)
            # Flatten to 1D array
            preds = outputs.cpu().numpy().flatten()

            ids_list.append(ids)
            preds_list.append(preds)

    all_ids = np.concatenate(ids_list)
    all_preds = np.concatenate(preds_list)

    # Create submission DataFrame
    df = pd.DataFrame({"id": all_ids, "target": all_preds})

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training_pipeline(
    epochs=20,
    batch_size=1024,
    learning_rate=1e-3,
    embedding_dim=32,
    hidden_units=[512, 256, 128],
    dropout_rate=0.2,
    patience=5,
    sample_size=None,
    load_cached_data=True,
):
    """
    Orchestrates the entire training and prediction pipeline.
    """
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets(
        load_cached_data=load_cached_data, sample_size=sample_size
    )

    # Create DataLoaders
    # num_workers=4 is safe given 12 vCPUs
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 2. Determine Model Dimensions dynamically
    sample_batch = next(iter(train_loader))
    num_numerical = sample_batch["numerical"].shape[1]
    seq_len = sample_batch["sequence"].shape[1]

    # Determine vocab size (max index + 1)
    max_idx_train = train_ds.X_seq.max()
    max_idx_val = val_ds.X_seq.max()
    max_idx_test = test_ds.X_seq.max()
    vocab_size = int(max(max_idx_train, max_idx_val, max_idx_test)) + 1

    print(
        f"Model Configuration: Num Features={num_numerical}, Seq Len={seq_len}, Vocab Size={vocab_size}"
    )

    # 3. Initialize Model
    model = ManufacturingNet(
        num_numerical_features=num_numerical,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        seq_len=seq_len,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
    ).to(device)

    # 4. Setup Optimization
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.5f} | Train AUC: {train_auc:.5f} | "
            f"Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.10f}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation AUC: {best_auc:.10f}")

    # 6. Generate Submission
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    generate_submission(model, test_loader, device)
