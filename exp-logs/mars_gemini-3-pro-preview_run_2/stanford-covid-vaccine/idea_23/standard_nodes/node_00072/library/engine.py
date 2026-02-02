import os
import torch
import torch.optim as optim
from library.config import Config
from library.utils import GlobalMCRMSE, set_seed
from library.loss import MaskedMCRMSELoss
from library.model import RNAModel
from library.data import get_dataloaders


def train_fn(model, dataloader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for inputs, partner_indices, targets, _ in dataloader:
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: Model expects (inputs, partner_indices)
        outputs = model(inputs, partner_indices)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def eval_fn(model, dataloader, device):
    """
    Executes validation using the Global MCRMSE metric.
    """
    model.eval()
    metric = GlobalMCRMSE()

    with torch.no_grad():
        for inputs, partner_indices, targets, _ in dataloader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            outputs = model(inputs, partner_indices)

            # Update global accumulator
            metric.update(outputs, targets)

    return metric.compute()


def run_training(debug=False):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Starting training on device: {device}")

    # 2. Data
    # get_dataloaders handles caching via get_data
    train_loader, val_loader, _ = get_dataloaders(debug=debug, load_cached_data=True)

    # 3. Model
    model = RNAModel()
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    # Reduce LR if validation score plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = MaskedMCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = eval_fn(model, val_loader, device)

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        # Scheduler Step
        scheduler.step(val_score)

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print(f"Training finished. Best Validation MCRMSE: {best_score}")
    return best_score
