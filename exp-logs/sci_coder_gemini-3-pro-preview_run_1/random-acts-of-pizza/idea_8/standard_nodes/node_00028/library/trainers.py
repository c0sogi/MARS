import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed
from library.architectures import GatedFusionMLP


def train_random_forest(X_train, y_train, X_val, y_val, X_test):
    """
    Trains a Random Forest classifier using the configuration parameters.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        X_test (np.ndarray): Test features.

    Returns:
        tuple: (val_preds, test_preds, model)
    """
    set_seed()
    print("Initializing Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        max_depth=Config.RF_MAX_DEPTH,
        min_samples_split=Config.RF_MIN_SAMPLES_SPLIT,
        class_weight="balanced",
        random_state=Config.SEED,
        n_jobs=-1,
    )

    print("Fitting Random Forest...")
    rf.fit(X_train, y_train)

    print("Generating predictions...")
    # Get probability of the positive class (1)
    val_preds = rf.predict_proba(X_val)[:, 1]
    test_preds = rf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_val, val_preds)
    print(f"Random Forest Validation AUC: {auc}")

    return val_preds, test_preds, rf


def train_neural_net(data, y_train, y_val):
    """
    Trains the GatedFusionMLP neural network with early stopping.

    Args:
        data (dict): Dictionary containing input features. Expected keys:
                     'X_train_text', 'X_train_comm', 'X_train_tab',
                     'X_val_text', 'X_val_comm', 'X_val_tab',
                     'X_test_text', 'X_test_comm', 'X_test_tab'
        y_train (np.ndarray): Training targets.
        y_val (np.ndarray): Validation targets.

    Returns:
        tuple: (val_preds, test_preds, model)
    """
    set_seed()
    print("Initializing Gated Fusion MLP Training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    # Convert numpy arrays to tensors
    X_train_text = torch.FloatTensor(data["X_train_text"]).to(device)
    X_train_comm = torch.FloatTensor(data["X_train_comm"]).to(device)
    X_train_tab = torch.FloatTensor(data["X_train_tab"]).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    X_val_text = torch.FloatTensor(data["X_val_text"]).to(device)
    X_val_comm = torch.FloatTensor(data["X_val_comm"]).to(device)
    X_val_tab = torch.FloatTensor(data["X_val_tab"]).to(device)
    # y_val is kept as numpy for metric calculation, but we need tensor for loss if we calculated val loss

    X_test_text = torch.FloatTensor(data["X_test_text"]).to(device)
    X_test_comm = torch.FloatTensor(data["X_test_comm"]).to(device)
    X_test_tab = torch.FloatTensor(data["X_test_tab"]).to(device)

    # Create DataLoader
    train_ds = TensorDataset(X_train_text, X_train_comm, X_train_tab, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=Config.MLP_BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    # Infer dimensions from input data
    text_dim = X_train_text.shape[1]
    comm_dim = X_train_comm.shape[1]
    tab_dim = X_train_tab.shape[1]

    model = GatedFusionMLP(
        text_dim=text_dim,
        comm_dim=comm_dim,
        tab_dim=tab_dim,
        hidden_dim=Config.MLP_HIDDEN_DIM,
        dropout=Config.MLP_DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.MLP_LEARNING_RATE,
        weight_decay=Config.MLP_WEIGHT_DECAY,
    )
    criterion = nn.BCELoss()

    # 3. Training Loop with Early Stopping
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(
        f"Starting training for {Config.MLP_EPOCHS} epochs with patience {Config.MLP_PATIENCE}..."
    )

    for epoch in range(Config.MLP_EPOCHS):
        model.train()
        train_loss = 0.0

        for b_text, b_comm, b_tab, b_y in train_loader:
            optimizer.zero_grad()
            preds = model(b_text, b_comm, b_tab)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Step
        model.eval()
        with torch.no_grad():
            val_preds_tensor = model(X_val_text, X_val_comm, X_val_tab)
            val_preds_np = val_preds_tensor.cpu().numpy().flatten()

        current_val_auc = roc_auc_score(y_val, val_preds_np)

        print(
            f"Epoch {epoch+1}/{Config.MLP_EPOCHS} - "
            f"Train Loss: {avg_train_loss:.6f} - "
            f"Val AUC: {current_val_auc}"
        )

        # Early Stopping Logic
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.MLP_PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_val_auc}"
            )
            break

    # 4. Restore Best Model and Inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No improvement during training. Using last model state.")

    model.eval()
    with torch.no_grad():
        final_val_preds = (
            model(X_val_text, X_val_comm, X_val_tab).cpu().numpy().flatten()
        )
        final_test_preds = (
            model(X_test_text, X_test_comm, X_test_tab).cpu().numpy().flatten()
        )

    return final_val_preds, final_test_preds, model


def predict_ensemble(rf_preds, mlp_preds, weights=(0.5, 0.5)):
    """
    Combines predictions from multiple models using weighted averaging.

    Args:
        rf_preds (np.ndarray): Predictions from Random Forest.
        mlp_preds (np.ndarray): Predictions from MLP.
        weights (tuple): Weights for (rf, mlp). Sum should ideally be 1.0.

    Returns:
        np.ndarray: Combined predictions.
    """
    print(f"Ensembling predictions with weights RF={weights[0]}, MLP={weights[1]}...")
    ensemble_preds = (weights[0] * rf_preds) + (weights[1] * mlp_preds)
    return ensemble_preds
