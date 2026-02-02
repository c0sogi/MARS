import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from library.utils import get_device, compute_log_loss, set_seed

# Constants
TEMP_MODEL_PATH = "./working/idea_3/temp_best_model.pt"


def train_classical_model(model_wrapper, X_train, y_train, X_val=None, y_val=None):
    """
    Trains a classical model (Logistic Regression, Naive Bayes, XGBoost) using the provided wrapper.
    Evaluates on validation set if provided.

    Args:
        model_wrapper: Instance of ClassicalModelWrapper.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).

    Returns:
        tuple: (trained_model, val_probs, val_loss)
               val_probs and val_loss are None if validation data is not provided.
    """
    print(f"Training classical model: {model_wrapper.model_type}")

    # Fit the model
    model_wrapper.fit(X_train, y_train)

    val_probs = None
    val_loss = None

    if X_val is not None and y_val is not None:
        # Predict probabilities
        val_probs = model_wrapper.predict_proba(X_val)

        # Compute loss
        val_loss = compute_log_loss(y_val, val_probs)
        print(f"Validation Log Loss: {val_loss}")

    return model_wrapper, val_probs, val_loss


def _create_dataloader(features, labels, batch_size, shuffle=True):
    """
    Helper to create a PyTorch DataLoader from feature dicts and labels.
    """
    input_ids = torch.tensor(features["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(features["attention_mask"], dtype=torch.long)

    if labels is not None:
        y = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(input_ids, attention_mask, y)
    else:
        dataset = TensorDataset(input_ids, attention_mask)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True
    )


def train_neural_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    """
    Executes one training epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Unpack batch and move to device
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer and scheduler steps
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def validate_neural(model, dataloader, device, loss_fn=None):
    """
    Evaluates the model on a validation set.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                all_labels.append(labels.cpu().numpy())
            else:
                input_ids, attention_mask = [b.to(device) for b in batch]
                labels = None

            logits = model(input_ids, attention_mask)

            if loss_fn is not None and labels is not None:
                loss = loss_fn(logits, labels)
                total_loss += loss.item() * input_ids.size(0)

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    avg_loss = None
    if loss_fn is not None and len(all_labels) > 0:
        avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss, all_preds


def train_neural_model(
    model, train_features, train_labels, val_features, val_labels, config
):
    """
    Orchestrates the full training loop for a neural model including:
    - Data loading
    - Optimization (AdamW + Schedule)
    - Early Stopping
    - Model Checkpointing

    Args:
        model: The PyTorch model (TransformerClassifier).
        train_features: Dict with 'input_ids' and 'attention_mask'.
        train_labels: Numpy array of labels.
        val_features: Dict with 'input_ids' and 'attention_mask'.
        val_labels: Numpy array of labels.
        config: Configuration dictionary.

    Returns:
        tuple: (best_model, val_probs, best_val_loss)
    """
    set_seed(config.get("seed", 42))
    device = get_device()
    model = model.to(device)

    # Hyperparameters
    batch_size = config.get("batch_size", 16)
    epochs = config.get("epochs", 5)
    lr = config.get("learning_rate", 2e-5)
    patience = config.get("patience", 2)

    # Prepare DataLoaders
    train_loader = _create_dataloader(
        train_features, train_labels, batch_size, shuffle=True
    )
    val_loader = _create_dataloader(val_features, val_labels, batch_size, shuffle=False)

    # Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Ensure working directory exists for temp saving
    os.makedirs(os.path.dirname(TEMP_MODEL_PATH), exist_ok=True)

    print(f"Starting neural training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_neural_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn
        )
        val_loss, val_probs = validate_neural(model, val_loader, device, loss_fn)

        # Calculate metric using the competition specific function for consistency
        # (Though CrossEntropyLoss is mathematically similar, compute_log_loss handles clipping/rescaling)
        metric_loss = compute_log_loss(val_labels, val_probs)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Metric: {metric_loss}"
        )

        # Early Stopping Logic
        if metric_loss < best_val_loss:
            best_val_loss = metric_loss
            patience_counter = 0
            # Save best state in memory (or disk if memory is tight, but here memory is 220GB)
            # We use disk to be safe against any reference issues
            torch.save(model.state_dict(), TEMP_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    if os.path.exists(TEMP_MODEL_PATH):
        model.load_state_dict(torch.load(TEMP_MODEL_PATH, map_location=device))
        # Clean up
        os.remove(TEMP_MODEL_PATH)

    # Generate final validation predictions with best model
    _, final_val_probs = validate_neural(model, val_loader, device)

    return model, final_val_probs, best_val_loss
