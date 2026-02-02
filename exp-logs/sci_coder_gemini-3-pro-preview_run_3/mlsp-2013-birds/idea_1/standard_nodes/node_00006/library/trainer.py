import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library import config, dataset, model


def compute_auc(y_true, y_pred):
    """
    Computes Macro-Average ROC AUC, handling classes with no positive samples.

    Args:
        y_true (np.ndarray): Binary ground truth matrix (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities (N, num_classes).

    Returns:
        float: Macro-averaged AUC score.
    """
    auc_scores = []
    # Iterate over each class
    for i in range(y_true.shape[1]):
        # Only calculate AUC if the class is present in y_true (contains both 0 and 1)
        # If a class has all 0s or all 1s, AUC is undefined.
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                pass

    if not auc_scores:
        return 0.0

    return np.mean(auc_scores)


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    auc_score = compute_auc(y_true, y_pred)

    return epoch_loss, auc_score


def predict(model, loader, device):
    """
    Generates predictions for the test set.

    Returns:
        dict: Mapping rec_id -> probability vector (numpy array)
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for inputs, rec_ids in loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            rec_ids_np = rec_ids.numpy()

            for rid, prob_vec in zip(rec_ids_np, probs_np):
                results[rid] = prob_vec

    return results


def save_submission(predictions, output_path):
    """
    Formats and saves the submission CSV.

    Args:
        predictions (dict): Mapping rec_id -> probability vector.
        output_path (str): Path to save the CSV.
    """
    data = []

    # Sort by rec_id to ensure order (though not strictly required, it's good practice)
    sorted_ids = sorted(predictions.keys())

    for rec_id in sorted_ids:
        probs = predictions[rec_id]
        for species_idx, prob in enumerate(probs):
            # Format: Id = rec_id * 100 + species_number
            submission_id = int(rec_id * 100 + species_idx)
            data.append({"Id": submission_id, "Probability": prob})

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run():
    """
    Main execution function.
    """
    # 1. Setup
    dataset.set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    net = model.ShallowMLP(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop with Early Stopping
    best_auc = -1.0
    patience_counter = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(net, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(net, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val AUC: {val_auc:.10f}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = net.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 5. Final Inference
    print(f"Training complete. Best Val AUC: {best_auc:.10f}")

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    print("Generating predictions on test set...")
    test_predictions = predict(net, test_loader, device)

    # 6. Save Submission
    save_submission(test_predictions, config.SUBMISSION_PATH)
