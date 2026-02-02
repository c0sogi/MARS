import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from library.config import Config, PizzaDataset, AttentionGatedMLP
from library.utils import set_seed


def train_nn_model(
    train_data,
    val_data,
    sub_emb,
    hidden_dim=Config.MLP_HIDDEN_DIM,
    dropout=Config.MLP_DROPOUT,
    lr=Config.MLP_LR,
    epochs=Config.MLP_EPOCHS,
    patience=Config.MLP_PATIENCE,
    batch_size=Config.MLP_BATCH_SIZE,
    seed=Config.SEED,
    device=None,
):
    """
    Trains the AttentionGatedMLP model with early stopping.

    Args:
        train_data (tuple): Tuple containing (title, body, hist, meta, y) for training.
        val_data (tuple): Tuple containing (title, body, hist, meta, y) for validation.
        sub_emb (np.ndarray): Pretrained subreddit embeddings matrix.
        hidden_dim (int): Hidden dimension size for the MLP.
        dropout (float): Dropout rate.
        lr (float): Learning rate.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        batch_size (int): Batch size for DataLoaders.
        seed (int): Random seed.
        device (torch.device, optional): Device to train on.

    Returns:
        model (nn.Module): The trained model with the best validation weights loaded.
    """
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack data
    # train_data structure: (title_emb, body_emb, hist_idx, meta, y)
    train_dataset = PizzaDataset(*train_data)
    val_dataset = PizzaDataset(*val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # input_dim for metadata is the last dimension of the meta array
    meta_dim = train_data[3].shape[1]

    # Initialize Model
    model = AttentionGatedMLP(
        sub_embeddings=sub_emb,
        meta_dim=meta_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            title = batch["title"].to(device)
            body = batch["body"].to(device)
            hist = batch["hist"].to(device)
            meta = batch["meta"].to(device)
            y = batch["y"].to(device)

            logits = model(title, body, hist, meta)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                title = batch["title"].to(device)
                body = batch["body"].to(device)
                hist = batch["hist"].to(device)
                meta = batch["meta"].to(device)
                y = batch["y"]  # Keep on CPU for sklearn metric

                logits = model(title, body, hist, meta)
                probs = torch.sigmoid(logits).cpu().numpy()

                val_preds.extend(probs)
                val_targets.extend(y.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        # Check Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc}"
            )
        else:
            patience_counter += 1
            if patience_counter % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | Val AUC: {val_auc} (No improvement for {patience_counter} epochs)"
                )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_nn_model(model, test_data, batch_size=Config.MLP_BATCH_SIZE, device=None):
    """
    Generates predictions using the trained NN model.

    Args:
        model (nn.Module): Trained AttentionGatedMLP model.
        test_data (tuple): Tuple containing (title, body, hist, meta) for testing.
        batch_size (int): Batch size.
        device (torch.device, optional): Device to run inference on.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test_data structure: (title_emb, body_emb, hist_idx, meta)
    # Note: PizzaDataset handles y=None gracefully
    test_dataset = PizzaDataset(*test_data, y=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            title = batch["title"].to(device)
            body = batch["body"].to(device)
            hist = batch["hist"].to(device)
            meta = batch["meta"].to(device)

            logits = model(title, body, hist, meta)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)

    return np.array(all_preds)
