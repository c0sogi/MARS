import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config, ECGRN, FocalLoss, ContactDataset, get_data, set_seed
from library.utils import compute_mcc


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Unpack batch; ContactDataset returns (X, y) when y is present
        X, y = batch
        X = X.to(device)
        y = y.to(device).unsqueeze(1)  # Reshape for BCE/Focal Loss

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Generates predictions on the validation set without updating gradients.
    Returns predicted probabilities and true labels.
    """
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                X, y = batch
                targets_list.append(y.numpy())
            else:
                X = batch

            X = X.to(device)
            preds = model(X)
            preds_list.append(preds.cpu().numpy())

    preds_arr = np.concatenate(preds_list)
    targets_arr = np.concatenate(targets_list) if targets_list else None

    return preds_arr, targets_arr


def optimize_threshold(y_true, y_probs):
    """
    Performs a grid search to find the decision threshold that maximizes MCC.
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Grid search from 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, 81)

    for t in thresholds:
        score = compute_mcc(y_true, y_probs, threshold=t)
        if score > best_mcc:
            best_mcc = score
            best_thresh = t

    return best_thresh, best_mcc


def run_training():
    """
    Orchestrates the training pipeline:
    1. Loads and splits data.
    2. Scales features.
    3. Initializes model, optimizer, and loss.
    4. Runs training loop with validation and early stopping.
    5. Optimizes threshold.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    # get_data handles caching and processing
    df_all = get_data(mode="train", load_cached_data=True)

    exclude_cols = ["contact_id", "game_play", "step", "contact", "is_val"]
    feature_cols = [c for c in df_all.columns if c not in exclude_cols]

    # Set input dimension dynamically
    Config.INPUT_DIM = len(feature_cols)
    print(f"Input Dimension: {Config.INPUT_DIM}")

    # 2. Split & Scale
    train_df = df_all[df_all["is_val"] == 0]
    val_df = df_all[df_all["is_val"] == 1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    y_train = train_df["contact"].values.astype(float)

    X_val = scaler.transform(val_df[feature_cols].values)
    y_val = val_df["contact"].values.astype(float)

    # Create Dataloaders
    train_loader = DataLoader(
        ContactDataset(X_train, y_train),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        ContactDataset(X_val, y_val),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # 3. Model Setup
    model = ECGRN(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        dropout=Config.DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # 4. Training Loop
    best_mcc = -1.0
    best_state = None
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_probs, val_targets = validate(model, val_loader, device)

        # Quick check with default threshold 0.5 for monitoring
        current_mcc = compute_mcc(val_targets, val_probs, threshold=0.5)

        # Print full precision as requested
        print(
            f"Epoch {epoch+1} | Loss: {train_loss:.6f} | Val MCC (0.5): {current_mcc:.16f}"
        )

        # Early Stopping Logic
        if current_mcc > best_mcc:
            best_mcc = current_mcc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # 5. Threshold Optimization
    if best_state is not None:
        model.load_state_dict(best_state)

    # Get predictions with best model
    val_probs, val_targets = validate(model, val_loader, device)

    best_thresh, best_mcc_opt = optimize_threshold(val_targets, val_probs)

    print(f"Best Threshold: {best_thresh} | Best Val MCC: {best_mcc_opt:.16f}")

    return model, scaler, best_thresh, feature_cols
