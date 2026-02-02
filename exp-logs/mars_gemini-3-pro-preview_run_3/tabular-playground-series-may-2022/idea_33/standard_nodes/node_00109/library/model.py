import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from library.utils import process_data, seed_everything, get_device


class CRHPEModel(nn.Module):
    def __init__(self, vocab_sizes, num_cont):
        super(CRHPEModel, self).__init__()
        self.num_streams = 5
        self.emb_dim = 16

        # Independent Embeddings per stream: List[List[Embedding]]
        # Outer list: Stream index
        # Inner list: Feature index
        self.embeddings = nn.ModuleList(
            [
                nn.ModuleList([nn.Embedding(v, self.emb_dim) for v in vocab_sizes])
                for _ in range(self.num_streams)
            ]
        )

        # Input dimension for MLP = (Num Cat * Emb Dim) + Num Cont
        input_dim = len(vocab_sizes) * self.emb_dim + num_cont

        # Deep Paths (Funnels)
        # Cite Lesson 71: Avoid streams deeper than 3 layers to prevent overfitting.
        # Cite Lesson 69: Use regularization heterogeneity (varying dropout) for diversity.
        # Cite Lesson 87: Remove linear residuals (Pure Deep architecture).
        self.mlps = nn.ModuleList()

        # Stream Configurations: (Width, Dropout)
        # We use a range of dropout [0.15, 0.25] (Cite Lesson 77)
        configs = [(512, 0.15), (512, 0.20), (512, 0.25), (1024, 0.15), (1024, 0.20)]

        for width, dropout in configs:
            # Reduced to 3 Linear layers (2 hidden) as per Lesson 71
            layers = [
                nn.Linear(input_dim, width),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(width, width // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(width // 2, 1),
            ]
            self.mlps.append(nn.Sequential(*layers))

    def forward(self, x_cat, x_cont):
        outputs = []
        for i in range(self.num_streams):
            # 1. Embeddings
            embs = [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings[i])]
            x_emb = torch.cat(embs, dim=1)

            # 2. Early Fusion
            x_fused = torch.cat([x_emb, x_cont], dim=1)

            # 3. Deep Path (Pure Deep, no Linear Residual)
            outputs.append(self.mlps[i](x_fused))

        return outputs


def train_and_predict(load_cached_data=True, epochs=50, batch_size=1024):
    seed_everything(42)
    device = get_device()

    # Load Data using library utility
    data = process_data(load_cached_data)
    X_cat_train, X_cont_train, y_train = data[0], data[1], data[2]
    X_cat_val, X_cont_val, y_val = data[3], data[4], data[5]
    X_cat_test, X_cont_test, test_ids = data[6], data[7], data[8]
    vocab_sizes = data[9]

    # Create DataLoaders
    train_dataset = TensorDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = TensorDataset(X_cat_val, X_cont_val, y_val)
    test_dataset = TensorDataset(X_cat_test, X_cont_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize Model
    model = CRHPEModel(vocab_sizes, X_cont_train.shape[1]).to(device)

    # Optimization
    # Cite Lesson 96: Reduce weight decay to 1e-5 to avoid underfitting continuous features
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_path = "./working/best_model.pth"

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for cat_batch, cont_batch, y_batch in train_loader:
            cat_batch = cat_batch.to(device)
            cont_batch = cont_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()

            # Forward pass returns list of outputs from 5 streams
            outputs = model(cat_batch, cont_batch)

            # Sum loss across all streams
            loss = 0
            for out in outputs:
                loss += criterion(out, y_batch)

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
            for cat_batch, cont_batch, y_batch in val_loader:
                cat_batch = cat_batch.to(device)
                cont_batch = cont_batch.to(device)
                outputs = model(cat_batch, cont_batch)

                # Ensemble averaging (Sigmoid -> Mean)
                probs = torch.zeros_like(outputs[0])
                for out in outputs:
                    probs += torch.sigmoid(out)
                probs /= len(outputs)

                val_preds.append(probs.cpu().numpy())
                val_targets.append(y_batch.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f} | Val AUC: {auc:.10f}"
        )

        # Checkpointing
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # Inference on Test Set
    print("Generating predictions on Test Set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_preds = []

    with torch.no_grad():
        for cat_batch, cont_batch in test_loader:
            cat_batch = cat_batch.to(device)
            cont_batch = cont_batch.to(device)
            outputs = model(cat_batch, cont_batch)

            probs = torch.zeros_like(outputs[0])
            for out in outputs:
                probs += torch.sigmoid(out)
            probs /= len(outputs)

            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()

    # Save Submission
    os.makedirs("./submission", exist_ok=True)
    sub_df = pd.DataFrame({"id": test_ids, "target": test_preds})
    sub_df.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")
