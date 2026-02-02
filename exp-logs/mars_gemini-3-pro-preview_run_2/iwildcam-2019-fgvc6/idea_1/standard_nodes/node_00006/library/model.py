import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from library.config import Config
from library.utils import set_seed, calculate_macro_f1


class LinearProbe(nn.Module):
    """
    A simple linear classifier.
    Input: Concatenated features (Global Avg Pool + Global Max Pool) -> 4096 dim
    Output: Class logits -> 23 dim
    """

    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_model(
    train_features,
    train_targets,
    val_features,
    val_targets,
    epochs=Config.EPOCHS,
    lr=Config.LEARNING_RATE,
    patience=5,
):
    """
    Trains the LinearProbe model using L-BFGS optimizer and class weighting.

    Args:
        train_features (np.ndarray): Training feature vectors.
        train_targets (np.ndarray): Training labels.
        val_features (np.ndarray): Validation feature vectors.
        val_targets (np.ndarray): Validation labels.
        epochs (int): Maximum number of epochs.
        lr (float): Learning rate for L-BFGS.
        patience (int): Early stopping patience.

    Returns:
        model: The trained PyTorch model with the best validation performance.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing training on {device}...")

    # Convert data to tensors
    # Use float() to ensure float32, long() for targets
    X_train = torch.from_numpy(train_features).float().to(device)
    y_train = torch.from_numpy(train_targets).long().to(device)
    X_val = torch.from_numpy(val_features).float().to(device)
    # y_val is kept as numpy for metric calculation to avoid unnecessary GPU-CPU transfers

    # 1. Handle Class Imbalance
    # Calculate weights inversely proportional to class frequencies
    # We assume classes 0..22 are the set.
    present_classes = np.unique(train_targets)
    class_weights_subset = compute_class_weight(
        class_weight="balanced", classes=present_classes, y=train_targets
    )

    # Initialize full weight vector with 1.0 (default for missing classes)
    class_weights_full = np.ones(Config.NUM_CLASSES, dtype=np.float32)
    class_weights_full[present_classes] = class_weights_subset

    class_weights = torch.tensor(class_weights_full, dtype=torch.float32).to(device)

    # 2. Initialize Model
    input_dim = X_train.shape[1]
    model = LinearProbe(input_dim, Config.NUM_CLASSES).to(device)

    # 3. Loss & Optimizer
    # CrossEntropyLoss combines LogSoftmax and NLLLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # L-BFGS is excellent for convex problems (linear layer) with full-batch training
    # Added line_search_fn="strong_wolfe" for better convergence and weight_decay for regularization
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
        weight_decay=1e-4,
    )

    # 4. Training Loop
    best_val_f1 = -1.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()

        # L-BFGS requires a closure function that clears gradients and returns loss
        def closure():
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            return loss

        # Perform optimization step
        loss = optimizer.step(closure)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

        val_f1 = calculate_macro_f1(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f} | Val Macro F1: {val_f1}"
        )

        # Early Stopping Check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Cite solution_lesson_node_00003: In-Memory Checkpointing Requires Deep Copies
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered. Best Val F1: {best_val_f1}")
                break

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def generate_submission(
    model, test_features, test_ids, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): Trained model.
        test_features (np.ndarray): Test feature vectors.
        test_ids (np.ndarray): Test image IDs.
        output_path (str): Path to save the submission CSV.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Generating predictions for test set...")
    model.eval()

    X_test = torch.from_numpy(test_features).float().to(device)

    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # Construct DataFrame
    df = pd.DataFrame({"Id": test_ids, "Category": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
