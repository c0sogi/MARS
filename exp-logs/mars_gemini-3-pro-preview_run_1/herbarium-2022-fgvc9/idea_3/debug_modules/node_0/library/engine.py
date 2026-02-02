import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from library.config import Config
from library.utils import MetricMonitor, get_class_mappings


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    """
    Performs one epoch of training.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (
        images,
        species_targets,
        genus_targets,
        family_targets,
    ) in enumerate(loader):
        images = images.to(device)
        species_targets = species_targets.to(device)
        genus_targets = genus_targets.to(device)
        family_targets = family_targets.to(device)

        targets = (species_targets, genus_targets, family_targets)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss, loss_dict = loss_fn(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        for name, val in loss_dict.items():
            metric_monitor.update(name, val, batch_size)

    return metric_monitor.metrics


def validate(model, loader, device, loss_fn):
    """
    Evaluates the model on the validation set.
    Returns average loss and Macro F1 score for species.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, species_targets, genus_targets, family_targets in loader:
            images = images.to(device)
            species_targets = species_targets.to(device)
            genus_targets = genus_targets.to(device)
            family_targets = family_targets.to(device)

            targets = (species_targets, genus_targets, family_targets)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss, loss_dict = loss_fn(outputs, targets)

            # Update metrics
            batch_size = images.size(0)
            metric_monitor.update("loss_total", loss.item(), batch_size)

            # Collect predictions for F1 score (Species head only)
            species_logits = outputs["species"]
            preds = torch.argmax(species_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(species_targets.cpu().numpy())

    # Calculate Macro F1
    f1 = f1_score(all_targets, all_preds, average="macro")
    avg_loss = metric_monitor.get_avg("loss_total")

    return avg_loss, f1


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    loss_fn,
    num_epochs,
    patience=3,
):
    """
    Runs the full training loop with early stopping.
    """
    best_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {num_epochs} epochs on device: {device}")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        train_loss = (
            train_metrics["loss_total"]["sum"] / train_metrics["loss_total"]["count"]
        )

        # Validate
        val_loss, val_f1 = validate(model, val_loader, device, loss_fn)

        # Print metrics (Full precision as requested)
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Macro F1: {val_f1}")

        # Scheduler Step (Cosine Annealing at end of epoch)
        if scheduler:
            scheduler.step()

        # Early Stopping and Checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1: {best_f1}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation F1: {best_f1}")
    return best_f1


def predict_and_submit(model, test_loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    Saves the submission file.
    """
    print("Starting inference with TTA...")

    # Load best model weights
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded best model weights.")
    else:
        print("Warning: Best model weights not found. Using current model weights.")

    model.eval()

    # Get mappings to convert indices back to category_id
    _, idx_to_class = get_class_mappings(load_cached_data=True)

    image_ids = []
    predictions = []

    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)

            # --- Test-Time Augmentation (TTA) ---
            # 1. Original Prediction
            outputs_orig = model(images)
            logits_orig = outputs_orig["species"]

            # 2. Flipped Prediction (Horizontal Flip)
            images_flipped = torch.flip(images, dims=[3])  # N, C, H, W -> flip W
            outputs_flipped = model(images_flipped)
            logits_flipped = outputs_flipped["species"]

            # 3. Average Logits
            avg_logits = (logits_orig + logits_flipped) / 2.0

            # Get predicted class index
            preds_idx = torch.argmax(avg_logits, dim=1).cpu().numpy()

            # Map back to category_id
            preds_cat_id = [idx_to_class[idx] for idx in preds_idx]

            image_ids.extend(img_ids)
            predictions.extend(preds_cat_id)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(submission_df.head())
