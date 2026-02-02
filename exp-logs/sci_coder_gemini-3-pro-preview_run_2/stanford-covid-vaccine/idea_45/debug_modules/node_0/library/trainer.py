import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from library.loss_metric import MCRMSELoss, GlobalMetricsTracker
from library.data_processor import get_dataloaders
from library.model_architecture import RDFRN

# Configuration
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one epoch of training using the iterative refinement strategy.

    Args:
        model: The RDFRN model.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function (MCRMSELoss).
        device: Torch device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass returns both initial (y1) and refined (y2) predictions
        # The model handles the feedback loop internally
        y1, y2 = model(inputs, partner_indices)

        # Loss Calculation: MCRMSE(y2) + 0.5 * MCRMSE(y1)
        # The criterion handles column slicing (scoring only specific columns) internally
        loss2 = criterion(y2, targets)
        loss1 = criterion(y1, targets)
        loss = loss2 + 0.5 * loss1

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    """
    Validates the model on the validation set.
    Only the final refined prediction (y2) is used for metrics.

    Args:
        model: The RDFRN model.
        loader: DataLoader for validation data.
        device: Torch device.

    Returns:
        dict: Dictionary containing validation metrics.
    """
    model.eval()
    tracker = GlobalMetricsTracker()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # We only care about the final refined output y2 for validation scoring
            _, y2 = model(inputs, partner_indices)

            tracker.update(y2, targets)

    return tracker.compute()


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.

    Returns:
        tuple: (predictions numpy array, list of IDs)
    """
    model.eval()
    preds_list = []
    ids_list = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            ids = batch["id"]

            _, y2 = model(inputs, partner_indices)

            preds_list.append(y2.cpu().numpy())
            ids_list.extend(ids)

    return np.concatenate(preds_list, axis=0), ids_list


def generate_submission(
    model, test_loader, device, output_path="./submission/submission.csv"
):
    """
    Generates and saves the submission CSV file in the required format.
    """
    print("Generating Submission...")
    test_preds, test_ids = predict_test(model, test_loader, device)

    # Format submission
    # Need to flatten predictions: id_seqpos
    submission_rows = []

    # Columns in the model output correspond to:
    # 0: reactivity
    # 1: deg_Mg_pH10
    # 2: deg_pH10
    # 3: deg_Mg_50C
    # 4: deg_50C

    for i, sample_id in enumerate(test_ids):
        sample_preds = test_preds[i]  # Shape: (107, 5)

        for pos in range(sample_preds.shape[0]):
            row_id = f"{sample_id}_{pos}"
            vals = sample_preds[pos]

            # Create row list
            row = [row_id] + vals.tolist()
            submission_rows.append(row)

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df = pd.DataFrame(submission_rows, columns=columns)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(sub_df)} rows.")


def run_training(epochs=15, batch_size=32, load_cached_data=True):
    """
    Main execution function to run the training pipeline.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    set_seed(SEED)
    print(f"Initializing RDF-RN Model on {DEVICE}...")

    # Load Data
    # get_dataloaders handles the caching logic internally via get_data
    # It will create ./working/idea_45/ cache files if they don't exist
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=2, load_cached_data=load_cached_data
    )

    model = RDFRN().to(DEVICE)
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_mcrmse = float("inf")
    patience_counter = 0
    early_stopping_patience = 6

    # Ensure working directory exists for model saving
    os.makedirs("./working", exist_ok=True)
    model_save_path = "./working/best_model.pth"

    print("Starting Training...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = validate(model, val_loader, DEVICE)
        val_mcrmse = val_metrics["mcrmse"]

        duration = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {duration:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        scheduler.step(val_mcrmse)

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            print(f"  New best model saved! MCRMSE: {best_mcrmse}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{early_stopping_patience}"
            )
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    total_time = time.time() - start_time
    print(
        f"Training finished in {total_time:.2f}s. Best Validation MCRMSE: {best_mcrmse}"
    )

    # Load best model for inference
    if os.path.exists(model_save_path):
        print("Loading best model for inference...")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    else:
        print("Warning: No best model file found. Using current model state.")

    # Generate Submission
    generate_submission(model, test_loader, DEVICE)
