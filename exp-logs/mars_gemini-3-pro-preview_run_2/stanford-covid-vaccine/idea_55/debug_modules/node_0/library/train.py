import os
import torch
import torch.optim as optim
from library.config import Config
from library.model import DDARN
from library.loss import MaskedMCRMSELoss
from library.data import get_dataloaders
from library.utils import set_seed, get_device, MetricTracker


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch with the Dual Direct-Access Recurrent Network strategy.
    """
    model.train()
    running_loss = 0.0

    for X, partners, y in loader:
        X = X.to(device)
        partners = partners.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # --- Pass 1: Zero Feedback ---
        # Initial prediction without any feedback
        pred_1 = model(X, partners, prev_pred=None)

        # --- Pass 2: Recycled Feedback ---
        # Detach gradient to stop backprop through the feedback generation process itself (optional but stable)
        # or keep it attached. The idea specifies: "Detach Gradients: R = Y_1.detach()"
        fb_input = pred_1.detach().clone()

        # Mask unscored positions (indices >= 68) to prevent noise/leakage
        # Shape: (N, 5, L)
        fb_input[:, :, Config.PRED_LEN :] = 0.0

        # Second prediction using the refined feedback
        pred_2 = model(X, partners, prev_pred=fb_input)

        # --- Loss Calculation ---
        # Loss is calculated on both passes to supervise the intermediate step
        loss1 = criterion(pred_1, y)
        loss2 = criterion(pred_2, y)

        # Weighted sum: Focus primarily on final output, but stabilize initial output
        loss = loss2 + 0.5 * loss1

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using the global MCRMSE metric.
    """
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for X, partners, y in loader:
            X = X.to(device)
            partners = partners.to(device)
            y = y.to(device)

            # --- Pass 1 ---
            pred_1 = model(X, partners, prev_pred=None)

            # --- Pass 2 ---
            fb_input = pred_1.clone()
            fb_input[:, :, Config.PRED_LEN :] = 0.0

            pred_2 = model(X, partners, prev_pred=fb_input)

            # Update metric tracker with final predictions
            tracker.update(pred_2, y)

    return tracker.result()


def run_training(num_epochs=Config.NUM_EPOCHS, load_cached=True):
    """
    Main function to run the training pipeline.
    """
    # Reproducibility
    set_seed()
    device = get_device()

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Data Loaders
    train_loader = get_dataloaders(mode="train", load_cached=load_cached)
    val_loader = get_dataloaders(mode="val", load_cached=load_cached)

    # Model Setup
    model = DDARN().to(device)

    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = MaskedMCRMSELoss()

    # Training Loop State
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training on {device} for {num_epochs} epochs.")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Logging (Full precision as requested)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss} | Val MCRMSE: {val_loss}")

        # Model Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"Training complete. Best Validation MCRMSE: {best_val_loss}")
