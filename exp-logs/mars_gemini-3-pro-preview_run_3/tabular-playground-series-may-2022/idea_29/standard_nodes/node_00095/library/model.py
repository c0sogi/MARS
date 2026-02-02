import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.config import (
    EMBED_DIM,
    DROPOUT_RATES,
    MAX_LR,
    WEIGHT_DECAY,
    EPOCHS,
    WORKING_DIR,
    SUBMISSION_DIR,
)
from library.utils import save_checkpoint


class HC_PFE_Stream(nn.Module):
    def __init__(
        self, vocab_sizes, cat_cols, n_cont, embed_dim, hidden_layers, dropout_rate
    ):
        super().__init__()
        # Independent embeddings for this stream
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_sizes[col], embed_dim) for col in cat_cols]
        )

        n_cat_flat = len(cat_cols) * embed_dim
        input_dim = n_cat_flat + n_cont

        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat_x, cont_x):
        # cat_x: (Batch, N_Cat)
        emb_list = []
        for i, emb in enumerate(self.embeddings):
            emb_list.append(emb(cat_x[:, i]))

        # Flatten and concat
        x_cat = torch.cat(emb_list, dim=1)
        x = torch.cat([x_cat, cont_x], dim=1)

        return self.mlp(x)


class HCPFE_Model(nn.Module):
    def __init__(self, meta):
        super().__init__()
        cat_cols = meta["cat_cols"]
        vocab_sizes = meta["vocab_sizes"]
        n_cont = len(meta["cont_cols"])

        # Stream Configurations
        # Streams 1-3: Standard (512 -> 256 -> 128)
        # Streams 4-5: Wide (1024 -> 512 -> 256)
        configs = [
            {"layers": [512, 256, 128], "dropout": DROPOUT_RATES[0]},
            {"layers": [512, 256, 128], "dropout": DROPOUT_RATES[1]},
            {"layers": [512, 256, 128], "dropout": DROPOUT_RATES[2]},
            {"layers": [1024, 512, 256], "dropout": DROPOUT_RATES[3]},
            {"layers": [1024, 512, 256], "dropout": DROPOUT_RATES[4]},
        ]

        self.streams = nn.ModuleList()
        for cfg in configs:
            self.streams.append(
                HC_PFE_Stream(
                    vocab_sizes,
                    cat_cols,
                    n_cont,
                    EMBED_DIM,
                    cfg["layers"],
                    cfg["dropout"],
                )
            )

    def forward(self, cat_x, cont_x):
        outputs = []
        for stream in self.streams:
            outputs.append(stream(cat_x, cont_x))
        return outputs


def train_model(train_loader, val_loader, meta, device, epochs=EPOCHS):
    model = HCPFE_Model(meta).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for cat_x, cont_x, targets in train_loader:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(cat_x, cont_x)

            # Sum of BCE losses across all streams
            loss = 0
            for out in outputs:
                loss += criterion(out, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for cat_x, cont_x, targets in val_loader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                targets = targets.to(device)

                outputs = model(cat_x, cont_x)

                # Average probabilities
                probs = [torch.sigmoid(out) for out in outputs]
                avg_prob = torch.mean(torch.stack(probs, dim=0), dim=0)

                all_preds.append(avg_prob.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        val_auc = roc_auc_score(all_targets, all_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.5f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Val AUC: {best_auc:.10f}")
    return best_model_path


def generate_submission(model_path, test_loader, test_ids, meta, device):
    print("Generating submission...")
    model = HCPFE_Model(meta).to(device)
    # Load model weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for cat_x, cont_x, _ in test_loader:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)

            outputs = model(cat_x, cont_x)

            # Average probabilities
            probs = [torch.sigmoid(out) for out in outputs]
            avg_prob = torch.mean(torch.stack(probs, dim=0), dim=0)

            all_preds.append(avg_prob.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    submission = pd.DataFrame({"id": test_ids, "target": all_preds})

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
