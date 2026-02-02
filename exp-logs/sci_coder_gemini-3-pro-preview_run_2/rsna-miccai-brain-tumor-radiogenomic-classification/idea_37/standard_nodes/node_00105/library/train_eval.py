import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Tuple, List

from library.config import Config
from library.utils import get_logger, seed_everything
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet

# Initialize logger
logger = get_logger("Train_Eval")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation set.
    Returns: (Average Loss, ROC AUC Score)
    """
    model.eval()
    running_loss = 0.0
    count = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Handle edge case where only one class is present in batch/subset
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    return avg_loss, auc_score


def predict_tta(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Generates predictions using Test-Time Augmentation (Original + HFlip + VFlip).
    Returns: Array of probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in loader:
            # Handle case where loader returns (inputs, labels) or just inputs
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]

            inputs = inputs.to(device)

            # 1. Original
            out_orig = model(inputs)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (last dim)
            inputs_h = torch.flip(inputs, dims=[-1])
            out_h = model(inputs_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (second to last dim)
            inputs_v = torch.flip(inputs, dims=[-2])
            out_v = model(inputs_v)
            prob_v = torch.sigmoid(out_v)

            # Average
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0
            all_preds.extend(avg_prob.cpu().numpy())

    return np.array(all_preds).flatten()


def run_training(debug_limit: int = Config.DEBUG_LIMIT, load_cached_data: bool = True):
    """
    Main driver for training the model with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Data Loading
    logger.info("Loading DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=load_cached_data, debug_limit=debug_limit
    )

    # 2. Model Initialization
    model = AsymmetricEfficientNet()
    model.to(device)

    # 3. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    logger.info("Starting training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.10f} - "
            f"Val Loss: {val_loss:.10f} - "
            f"Val AUC: {val_auc:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            logger.info(f"New best model saved with AUC: {best_auc:.10f}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Val AUC: {best_auc:.10f}")


def generate_submission(
    load_cached_data: bool = True, debug_limit: int = Config.DEBUG_LIMIT
):
    """
    Loads the best model, runs TTA inference on the test set, and generates submission.csv.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, debug_limit=debug_limit
    )

    # 2. Load Model
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

    model = AsymmetricEfficientNet()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    logger.info("Best model loaded for inference.")

    # 3. Run Inference (TTA)
    logger.info("Running inference with TTA...")
    predictions = predict_tta(model, test_loader, device)

    # 4. Prepare Submission
    # Load metadata to map predictions to IDs
    # Note: The test_loader is created with shuffle=False, so order matches metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug_limit:
        df_test = df_test.head(debug_limit)

    # Align with valid IDs from cache if available
    # Cite debug_lesson_4: Validate Cache Schema Before Consumption
    if os.path.exists(Config.CACHE_TEST_IDS):
        valid_ids = np.load(Config.CACHE_TEST_IDS)
        # Filter and reorder metadata to match the cached data order
        df_test = df_test.set_index("BraTS21ID").loc[valid_ids].reset_index()

    if len(df_test) != len(predictions):
        logger.error(
            f"Mismatch: Metadata has {len(df_test)} rows, Predictions has {len(predictions)}"
        )
        # If mismatch persists, we align by length to prevent crash, though this implies logic error
        min_len = min(len(df_test), len(predictions))
        df_test = df_test.iloc[:min_len]
        predictions = predictions[:min_len]

    submission_df = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": predictions}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
