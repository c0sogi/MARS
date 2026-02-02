import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.dataset import prepare_dataloaders
from library.model import GRVCNet
from library.loss import FocalLoss
from library.feature_engineering import generate_dataset


def set_seed(seed):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0

    for batch in loader:
        # Unpack batch: dataset returns x_kin, x_vis, y
        x_kin, x_vis, y = batch
        x_kin, x_vis, y = x_kin.to(device), x_vis.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x_kin, x_vis)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluates the model on a given loader."""
    model.eval()
    total_loss = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            # Handle test loader which might not return targets
            if len(batch) == 3:
                x_kin, x_vis, y = batch
                y = y.to(device)
            else:
                x_kin, x_vis = batch
                y = None

            x_kin, x_vis = x_kin.to(device), x_vis.to(device)

            logits = model(x_kin, x_vis)

            if y is not None:
                loss = criterion(logits, y)
                total_loss += loss.item()
                all_targets.append(y.cpu().numpy())

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs)

    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        avg_loss = total_loss / len(loader)
        # Calculate MCC at default 0.5 threshold for monitoring
        mcc = matthews_corrcoef(all_targets, (all_probs > 0.5).astype(int))
        return avg_loss, mcc, all_probs, all_targets
    else:
        return 0.0, 0.0, all_probs, None


def optimize_threshold(targets, probs):
    """Finds the best threshold to maximize MCC."""
    best_thresh = 0.5
    best_mcc = -1
    thresholds = np.arange(0.1, 0.91, 0.01)

    for t in thresholds:
        preds = (probs > t).astype(int)
        mcc = matthews_corrcoef(targets, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    return best_thresh, best_mcc


def run_training(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS
):
    """
    Main function to run the training pipeline, optimization, and submission generation.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    # prepare_dataloaders handles caching and splitting internally via generate_dataset
    train_loader, val_loader, test_loader, dims = prepare_dataloaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )
    kin_dim, vis_dim = dims

    # Initialize Model
    model = GRVCNet(kin_dim, vis_dim, Config).to(device)

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # Training Loop
    best_val_mcc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mcc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}: Train Loss {train_loss:.10f}, Val Loss {val_loss:.10f}, Val MCC {val_mcc:.10f}"
        )

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model
    if os.path.exists(best_model_path):
        print("Loading best model for evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Threshold Optimization
    print("Optimizing threshold on validation set...")
    _, _, val_probs, val_targets = evaluate(model, val_loader, criterion, device)
    best_thresh, best_mcc = optimize_threshold(val_targets, val_probs)
    print(f"Best Threshold: {best_thresh:.4f}, Best Validation MCC: {best_mcc:.10f}")

    # Inference on Test Set
    print("Running inference on test set...")
    _, _, test_probs, _ = evaluate(model, test_loader, criterion, device)

    binary_preds = (test_probs > best_thresh).astype(int)

    # Generate Submission
    # We need contact_ids to map predictions. We load the test dataframe.
    # generate_dataset is deterministic and aligns with the test_loader order.
    df_test = generate_dataset(mode="test", load_cached_data=load_cached_data)

    submission = pd.DataFrame(
        {"contact_id": df_test["contact_id"], "contact": binary_preds}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
