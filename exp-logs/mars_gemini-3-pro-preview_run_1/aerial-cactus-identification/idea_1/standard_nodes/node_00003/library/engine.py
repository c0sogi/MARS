import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import Config, set_seed
from library.dataset import get_dataloaders
from library.model import CactusResNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the training set.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)  # Reshape to (Batch, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the validation set.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    # Cite solution_lesson_node_00002: Implementing 4-view TTA (Original, H-Flip, V-Flip, HV-Flip)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Generate 4 views
            inputs_h = torch.flip(inputs, [3])
            inputs_v = torch.flip(inputs, [2])
            inputs_hv = torch.flip(inputs, [2, 3])

            # Forward pass for all views
            out = model(inputs)
            out_h = model(inputs_h)
            out_v = model(inputs_v)
            out_hv = model(inputs_hv)

            # Average loss across views
            loss = (
                criterion(out, labels)
                + criterion(out_h, labels)
                + criterion(out_v, labels)
                + criterion(out_hv, labels)
            ) / 4.0

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Average probabilities
            probs = (
                torch.sigmoid(out)
                + torch.sigmoid(out_h)
                + torch.sigmoid(out_v)
                + torch.sigmoid(out_hv)
            ) / 4.0

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate ROC AUC
    # Handle edge cases (e.g., during debugging with small batches)
    if len(np.unique(all_targets)) > 1:
        auc_score = roc_auc_score(all_targets, all_preds)
    else:
        auc_score = 0.5

    return avg_loss, auc_score


def predict(model, dataloader, device, output_path, metadata_path, debug_size=None):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained neural network model.
        dataloader: DataLoader for the test set.
        device: Device to run inference on.
        output_path: Path to save the submission CSV.
        metadata_path: Path to the test metadata CSV (to retrieve IDs).
        debug_size: Optional limit on the number of samples (for debugging).
    """
    model.eval()
    preds = []

    # Cite solution_lesson_node_00002: Implementing 4-view TTA for inference
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            # Generate 4 views
            inputs_h = torch.flip(inputs, [3])
            inputs_v = torch.flip(inputs, [2])
            inputs_hv = torch.flip(inputs, [2, 3])

            # Forward pass
            out = model(inputs)
            out_h = model(inputs_h)
            out_v = model(inputs_v)
            out_hv = model(inputs_hv)

            # Average probabilities
            probs = (
                torch.sigmoid(out)
                + torch.sigmoid(out_h)
                + torch.sigmoid(out_v)
                + torch.sigmoid(out_hv)
            ) / 4.0

            preds.append(probs.cpu().numpy())

    # Flatten predictions
    all_preds = np.concatenate(preds).flatten()

    # Load metadata to get corresponding IDs
    df_test = pd.read_csv(metadata_path)
    if debug_size is not None:
        df_test = df_test.iloc[:debug_size]

    # Validate alignment
    if len(df_test) != len(all_preds):
        raise ValueError(
            f"Mismatch between metadata rows ({len(df_test)}) and predictions ({len(all_preds)})"
        )

    # Create submission DataFrame
    submission = pd.DataFrame({"id": df_test["id"], "has_cactus": all_preds})

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run(config: Config):
    """
    Main execution pipeline.
    1. Sets random seeds.
    2. Loads data.
    3. Initializes model, optimizer, and scheduler.
    4. Runs training loop with Early Stopping.
    5. Generates submission using the best model.

    Args:
        config: Configuration object.
    """
    set_seed(config.SEED)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # 2. Initialize Model
    model = CactusResNet(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    # 3. Optimization
    # BCEWithLogitsLoss is more stable than BCELoss + Sigmoid
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # Cosine Annealing for smooth learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.CACHE_DIR, "best_model.pth")

    print(f"Starting training for {config.NUM_EPOCHS} epochs on {config.DEVICE}...")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, config.DEVICE)

        scheduler.step()

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # 5. Prediction
    print("Loading best model for prediction...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    else:
        print("Warning: No best model found. Using current model weights.")

    predict(
        model=model,
        dataloader=test_loader,
        device=config.DEVICE,
        output_path=config.SUBMISSION_PATH,
        metadata_path=config.TEST_METADATA_PATH,
        debug_size=config.DEBUG_SUBSET_SIZE,
    )
