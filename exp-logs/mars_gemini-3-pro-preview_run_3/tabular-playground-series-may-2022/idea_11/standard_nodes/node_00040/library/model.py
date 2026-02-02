import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from library.config import Config


class TokenDropout(nn.Module):
    """
    Randomly zeros out entire embedding vectors in a sequence during training.
    This forces the model to learn distributed representations across the sequence.
    """

    def __init__(self, p=0.1):
        super(TokenDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x

        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # Generate mask: (batch_size, seq_len, 1)
        # We want to keep the token with probability (1 - p)
        keep_prob = 1 - self.p
        mask = torch.bernoulli(
            torch.full((batch_size, seq_len, 1), keep_prob, device=x.device)
        )

        # Apply mask and scale (Inverted Dropout)
        return (x * mask) / keep_prob


class LayerNormBlock(nn.Module):
    """
    A dense block consisting of Linear -> [LayerNorm] -> ReLU -> Dropout.
    LayerNorm is optional based on Config.
    """

    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(LayerNormBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.use_ln = Config.USE_LAYER_NORM
        if self.use_ln:
            self.ln = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        if self.use_ln:
            x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class LayerNormFunnelMLP(nn.Module):
    """
    Main architecture:
    1. Entity Embeddings for categorical features.
    2. Token Dropout applied specifically to f_27 character sequence.
    3. Early Fusion of flattened embeddings and continuous features.
    4. Funnel MLP Backbone (decreasing width) with Layer Normalization.
    """

    def __init__(
        self,
        vocab_sizes,
        cont_dim,
        embed_dim=None,
        hidden_layers=None,
        token_dropout_rate=None,
        dropout_rate=None,
    ):
        super(LayerNormFunnelMLP, self).__init__()

        # Use defaults from Config if not provided
        if embed_dim is None:
            embed_dim = Config.EMBEDDING_DIM
        if hidden_layers is None:
            hidden_layers = Config.HIDDEN_LAYERS
        if token_dropout_rate is None:
            token_dropout_rate = Config.TOKEN_DROPOUT_RATE
        if dropout_rate is None:
            dropout_rate = Config.DROPOUT_RATE

        self.vocab_sizes = vocab_sizes
        self.cont_dim = cont_dim
        self.embed_dim = embed_dim

        # Initialize Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # Token Dropout for the sequence of characters in f_27
        self.token_dropout = TokenDropout(p=token_dropout_rate)

        # Determine the start index of f_27 characters in the categorical list.
        # Config.CATEGORICAL_COLS = ["f_29", "f_30"], so f_27 chars start at index 2.
        self.f27_start_idx = len(Config.CATEGORICAL_COLS)
        self.num_cat = len(vocab_sizes)

        # Calculate input dimension for the MLP backbone
        # All categorical features are embedded to embed_dim
        total_cat_dim = self.num_cat * embed_dim
        input_dim = total_cat_dim + cont_dim

        # Build Funnel Backbone
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(LayerNormBlock(in_dim, h_dim, dropout_rate))
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # Output Head (Linear -> Logits)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x_cat, x_cont):
        # x_cat: (batch, num_cat) - LongTensor
        # x_cont: (batch, num_cont) - FloatTensor

        batch_size = x_cat.size(0)

        # 1. Look up all embeddings
        embed_list = []
        for i, emb_layer in enumerate(self.embeddings):
            embed_list.append(emb_layer(x_cat[:, i]))

        # 2. Process f_27 embeddings with Token Dropout
        # f_27 chars are located from self.f27_start_idx to the end
        f27_embeds = embed_list[self.f27_start_idx :]

        # Stack to shape (batch, 10, embed_dim)
        f27_stack = torch.stack(f27_embeds, dim=1)

        # Apply Token Dropout
        f27_stack = self.token_dropout(f27_stack)

        # Flatten back to (batch, 10 * embed_dim)
        f27_flat = f27_stack.view(batch_size, -1)

        # 3. Process other categorical embeddings (f_29, f_30)
        other_embeds = embed_list[: self.f27_start_idx]
        if other_embeds:
            other_flat = torch.cat(other_embeds, dim=1)
            # Concatenate all categorical parts
            cat_features = torch.cat([other_flat, f27_flat], dim=1)
        else:
            cat_features = f27_flat

        # 4. Early Fusion: Concatenate with continuous features
        x = torch.cat([cat_features, x_cont], dim=1)

        # 5. Pass through Backbone
        x = self.mlp(x)

        # 6. Output Head
        logits = self.head(x)

        return logits


def train_model(model, train_loader, val_loader, epochs=None, lr=None, device=None):
    """
    Executes the training loop with AdamW, OneCycleLR, and Early Stopping.
    """
    if epochs is None:
        epochs = Config.EPOCHS
    if lr is None:
        lr = Config.LEARNING_RATE
    if device is None:
        device = Config.DEVICE

    # Ensure save directory exists
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    model.to(device)

    # Optimizer: AdamW with calibrated weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: OneCycleLR for super-convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch_cat, batch_cont, batch_y in train_loader:
            batch_cat = batch_cat.to(device)
            batch_cont = batch_cont.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_cat, batch_cont)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_cat, batch_cont, batch_y in val_loader:
                batch_cat = batch_cat.to(device)
                batch_cont = batch_cont.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_cat, batch_cont)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                # Apply sigmoid for AUC calculation
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        val_auc = roc_auc_score(all_targets, all_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val AUC: {val_auc:.5f}"
        )

        # --- Early Stopping ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val AUC: {best_val_auc:.5f}")

    # Reload best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict(model, test_loader, device=None):
    """
    Generates probabilities for the test set and returns a DataFrame suitable for submission.
    """
    if device is None:
        device = Config.DEVICE

    model.eval()
    model.to(device)

    ids = []
    preds = []

    with torch.no_grad():
        for batch_cat, batch_cont, batch_ids in test_loader:
            batch_cat = batch_cat.to(device)
            batch_cont = batch_cont.to(device)

            logits = model(batch_cat, batch_cont)
            probs = torch.sigmoid(logits)

            ids.extend(batch_ids.numpy())
            preds.extend(probs.cpu().numpy().flatten())

    return pd.DataFrame({"id": ids, "target": preds})
