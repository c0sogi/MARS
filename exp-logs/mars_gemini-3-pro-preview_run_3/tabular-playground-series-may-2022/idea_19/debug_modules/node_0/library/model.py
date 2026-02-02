import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from library.config import Config


class HPFEModel(nn.Module):
    """
    Heterogeneous Parallel Funnel Ensemble (HPFE) Model.

    Consists of:
    - Shared Embedding Layer for categorical features.
    - Early Fusion of embeddings and continuous features.
    - 5 Parallel Independent MLP Streams with varying architectures and dropout rates.
    """

    def __init__(self, metadata):
        super().__init__()
        self.cat_cols = metadata["cat_cols"]
        self.cont_cols = metadata["cont_cols"]
        self.vocab_sizes = metadata["vocab_sizes"]

        # 1. Shared Embeddings
        self.embeddings = nn.ModuleList()
        for col in self.cat_cols:
            vocab_size = self.vocab_sizes[col]
            self.embeddings.append(nn.Embedding(vocab_size, Config.EMBEDDING_DIM))

        # Calculate Input Dimension for MLPs
        # Continuous features + (Number of Categorical Features * Embedding Dim)
        input_dim = len(self.cont_cols) + len(self.cat_cols) * Config.EMBEDDING_DIM

        # 2. Parallel Streams
        self.streams = nn.ModuleList()
        for conf in Config.STREAMS_CONFIG:
            layers = []
            prev_dim = input_dim

            # Build Hidden Layers
            for h_dim in conf["hidden_dims"]:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(conf["dropout"]))
                prev_dim = h_dim

            # Final Projection to Logit
            layers.append(nn.Linear(prev_dim, 1))

            self.streams.append(nn.Sequential(*layers))

    def forward(self, cont_x, cat_x):
        """
        Forward pass.

        Args:
            cont_x: Continuous features (Batch, Num_Cont)
            cat_x: Categorical features (Batch, Num_Cat)

        Returns:
            logits: List of 5 tensors, each (Batch, 1)
        """
        # Process Embeddings
        emb_list = []
        for i in range(len(self.cat_cols)):
            # Select column i, result is (Batch, Emb_Dim)
            emb_list.append(self.embeddings[i](cat_x[:, i]))

        # Flatten and Concatenate Embeddings
        x_cat = torch.cat(emb_list, dim=1)

        # Early Fusion: Concatenate Continuous and Categorical
        x = torch.cat([cont_x, x_cat], dim=1)

        # Pass through each stream
        logits = []
        for stream in self.streams:
            logits.append(stream(x))

        return logits


def train_model(model, train_loader, val_loader):
    """
    Trains the HPFE model using the specified strategy:
    - Optimizer: Adam
    - Scheduler: OneCycleLR
    - Loss: Sum of BCE from all streams
    - Early Stopping based on Val AUC
    """
    device = Config.DEVICE
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.MAX_EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    best_auc = 0.0
    patience_counter = 0
    best_model_path = Config.MODEL_SAVE_PATH

    print(
        f"Starting training on {device} with {len(train_loader)} batches per epoch..."
    )

    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            cont_x = batch["cont"].to(device)
            cat_x = batch["cat"].to(device)
            target = batch["target"].to(device).unsqueeze(1)  # (Batch, 1)

            optimizer.zero_grad()

            # Forward pass returns list of logits
            logits_list = model(cont_x, cat_x)

            # Compute sum of losses
            loss = 0
            for logits in logits_list:
                loss += criterion(logits, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                cont_x = batch["cont"].to(device)
                cat_x = batch["cat"].to(device)
                target = batch["target"].to(device)

                logits_list = model(cont_x, cat_x)

                # Ensemble Averaging (Mean of Sigmoids)
                probs = torch.zeros_like(logits_list[0])
                for logits in logits_list:
                    probs += torch.sigmoid(logits)
                probs /= len(logits_list)

                val_preds.append(probs.cpu().numpy())
                val_targets.append(target.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val AUC: {best_auc:.6f}")

    # Reload best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def predict_and_submit(model, test_loader):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = Config.DEVICE
    model.to(device)
    model.eval()

    all_preds = []
    all_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            cont_x = batch["cont"].to(device)
            cat_x = batch["cat"].to(device)
            ids = batch["id"].numpy()

            logits_list = model(cont_x, cat_x)

            # Ensemble Averaging
            probs = torch.zeros_like(logits_list[0])
            for logits in logits_list:
                probs += torch.sigmoid(logits)
            probs /= len(logits_list)

            all_preds.append(probs.cpu().numpy())
            all_ids.append(ids)

    all_preds = np.concatenate(all_preds).flatten()
    all_ids = np.concatenate(all_ids).flatten()

    # Create submission dataframe
    submission = pd.DataFrame({Config.ID_COL: all_ids, Config.TARGET_COL: all_preds})

    # Ensure ID format matches sample submission (int)
    submission[Config.ID_COL] = submission[Config.ID_COL].astype(int)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
