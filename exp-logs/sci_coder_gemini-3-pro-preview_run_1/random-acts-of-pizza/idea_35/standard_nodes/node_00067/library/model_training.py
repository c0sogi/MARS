import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, get_device
from library.model_architecture import DualQueryMLP, PizzaDataset


def train_rf(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators=Config.RF_ESTIMATORS,
    min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
    class_weight=Config.RF_CLASS_WEIGHT,
    random_state=Config.SEED,
    n_jobs=-1,
    debug_sample_size=None,
):
    """
    Trains a Random Forest Classifier.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        n_estimators (int): Number of trees.
        min_samples_leaf (int): Minimum samples per leaf.
        class_weight (str/dict): Class weight strategy.
        random_state (int): Random seed.
        n_jobs (int): Number of parallel jobs.
        debug_sample_size (int, optional): Limit training size for debugging.

    Returns:
        model: Trained RandomForestClassifier.
    """
    print("Initializing Random Forest training...")

    # Debugging: Subsample if requested
    if debug_sample_size is not None and debug_sample_size < len(X_train):
        print(f"Debugging: Subsampling training data to {debug_sample_size} samples.")
        indices = np.random.choice(len(X_train), debug_sample_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )

    print(f"Fitting Random Forest with {n_estimators} estimators...")
    rf.fit(X_train, y_train)

    # Validation
    print("Evaluating Random Forest on validation set...")
    val_preds = rf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)

    print(f"RF Validation AUC: {val_auc}")

    return rf


def predict_rf(model, X_test):
    """
    Generates predictions using the trained Random Forest model.

    Args:
        model: Trained RandomForestClassifier.
        X_test (np.ndarray): Test features.

    Returns:
        np.ndarray: Predicted probabilities for the positive class.
    """
    return model.predict_proba(X_test)[:, 1]


def train_mlp(
    train_data,
    val_data,
    hidden_dim=Config.MLP_HIDDEN_DIM,
    dropout=Config.MLP_DROPOUT,
    lr=Config.MLP_LR,
    weight_decay=Config.MLP_WEIGHT_DECAY,
    batch_size=Config.MLP_BATCH_SIZE,
    epochs=Config.MLP_EPOCHS,
    patience=Config.MLP_PATIENCE,
    debug_sample_size=None,
):
    """
    Trains the Dual-Query MLP with Early Stopping.

    Args:
        train_data (tuple): (title, body, history, meta, y) for training.
        val_data (tuple): (title, body, history, meta, y) for validation.
        hidden_dim (int): Hidden layer dimension.
        dropout (float): Dropout rate.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for AdamW.
        batch_size (int): Batch size.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        debug_sample_size (int, optional): Limit training size for debugging.

    Returns:
        model: Trained DualQueryMLP model (best state).
    """
    print("Initializing MLP training...")
    set_seed(Config.SEED)
    device = get_device()

    # Unpack data
    train_t, train_b, train_h, train_m, train_y = train_data
    val_t, val_b, val_h, val_m, val_y = val_data

    # Debugging: Subsample
    if debug_sample_size is not None and debug_sample_size < len(train_t):
        print(
            f"Debugging: Subsampling MLP training data to {debug_sample_size} samples."
        )
        indices = np.random.choice(len(train_t), debug_sample_size, replace=False)
        train_t = train_t[indices]
        train_b = train_b[indices]
        train_h = train_h[indices]
        train_m = train_m[indices]
        train_y = train_y[indices]

    # Create Datasets and Loaders
    train_ds = PizzaDataset(train_t, train_b, train_h, train_m, train_y)
    val_ds = PizzaDataset(val_t, val_b, val_h, val_m, val_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize Model
    emb_dim = train_t.shape[1]
    meta_dim = train_m.shape[1]

    model = DualQueryMLP(emb_dim, meta_dim, hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = -1.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # Move to device
            title = batch["title"].to(device)
            body = batch["body"].to(device)
            history = batch["history"].to(device)
            meta = batch["meta"].to(device)
            y = batch["y"].to(device)

            # Forward
            logits = model(title, body, history, meta)
            loss = criterion(logits, y)

            # Backward
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
                history = batch["history"].to(device)
                meta = batch["meta"].to(device)
                y = batch["y"]

                logits = model(title, body, history, meta)
                probs = torch.sigmoid(logits)

                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(y.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss} | Val AUC: {val_auc}")

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_auc}"
                )
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_mlp(model, test_data, batch_size=Config.MLP_BATCH_SIZE):
    """
    Generates predictions using the trained MLP model.

    Args:
        model: Trained DualQueryMLP model.
        test_data (tuple): (title, body, history, meta, ids) or (title, body, history, meta).
                           The last element is ignored if it's IDs/labels.
        batch_size (int): Batch size for inference.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    device = get_device()
    model.to(device)
    model.eval()

    # Unpack data - handle potential ID/Label at the end
    # We assume the first 4 elements are features
    test_t, test_b, test_h, test_m = (
        test_data[0],
        test_data[1],
        test_data[2],
        test_data[3],
    )

    test_ds = PizzaDataset(test_t, test_b, test_h, test_m)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    preds = []

    with torch.no_grad():
        for batch in test_loader:
            title = batch["title"].to(device)
            body = batch["body"].to(device)
            history = batch["history"].to(device)
            meta = batch["meta"].to(device)

            logits = model(title, body, history, meta)
            probs = torch.sigmoid(logits)

            preds.extend(probs.cpu().numpy())

    return np.array(preds)
