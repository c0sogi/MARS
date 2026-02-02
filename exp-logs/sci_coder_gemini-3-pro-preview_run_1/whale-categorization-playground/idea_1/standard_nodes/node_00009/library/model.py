import torch
import torch.nn as nn
import torchvision.models as models
import os
import pandas as pd
import numpy as np
import time
from library.utils import AverageMeter, calculate_map5, save_checkpoint, set_seed


class WhaleClassifier(nn.Module):
    """
    Whale species classifier based on ResNet-18.
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of unique whale IDs (including new_whale).
        """
        super(WhaleClassifier, self).__init__()
        # Load pre-trained ResNet-18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).
        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        return self.model(x)


def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    device,
    epoch,
    print_freq=10,
    max_batches=None,
):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, targets) in enumerate(train_loader):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

        if (i + 1) % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{i+1}/{len(train_loader)}] Loss {losses.val:.6f} ({losses.avg:.6f})"
            )

    return losses.avg


def validate(val_loader, model, criterion, device, max_batches=None):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(device)
            targets = targets.to(device)

            # Original forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), images.size(0))

            # TTA: Horizontal Flip
            # Average logits from original and flipped images for robust prediction
            outputs_flip = model(torch.flip(images, dims=[3]))
            outputs_tta = (outputs + outputs_flip) / 2.0

            # Get top 5 predictions for MAP@5 calculation
            # outputs is (Batch, NumClasses)
            _, preds = outputs_tta.topk(5, 1, True, True)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        map5 = calculate_map5(all_preds, all_targets)
    else:
        map5 = 0.0

    print(f"Validation: Loss {losses.avg:.6f} MAP@5 {map5:.10f}")
    return losses.avg, map5


def train_whale_model(
    train_loader,
    val_loader,
    num_classes,
    epochs=10,
    device="cuda",
    checkpoint_dir="./working",
    patience=3,
    lr=1e-4,
    max_batches_per_epoch=None,
):
    """
    Main training loop with early stopping.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_classes (int): Total number of classes.
        epochs (int): Maximum number of epochs.
        device (str): Device to train on ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save checkpoints.
        patience (int): Number of epochs with no improvement to wait before stopping.
        lr (float): Learning rate.
        max_batches_per_epoch (int, optional): Limit batches per epoch for debugging.

    Returns:
        model: The best trained model.
    """
    set_seed(42)

    model = WhaleClassifier(num_classes).to(device)
    # Use label smoothing to prevent overfitting on singleton classes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_map5 = -1.0
    epochs_no_improve = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            epoch + 1,
            max_batches=max_batches_per_epoch,
        )

        scheduler.step()

        # Validate
        val_loss, val_map5 = validate(
            val_loader, model, criterion, device, max_batches=max_batches_per_epoch
        )

        # Check for improvement
        is_best = val_map5 > best_map5
        if is_best:
            best_map5 = val_map5
            epochs_no_improve = 0
            print(f"New best MAP@5: {best_map5:.10f}")
        else:
            epochs_no_improve += 1
            print(
                f"No improvement. Best MAP@5: {best_map5:.10f}. Patience: {epochs_no_improve}/{patience}"
            )

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_score": best_map5,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            checkpoint_dir=checkpoint_dir,
        )

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load best model weights before returning
    best_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
    if os.path.exists(best_path):
        print(f"Loading best model from {best_path}")
        checkpoint = torch.load(best_path, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    return model


def generate_predictions(
    model, test_loader, label_encoder, device, output_file="./submission/submission.csv"
):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model: Trained WhaleClassifier.
        test_loader: DataLoader for test data (returns image, filename).
        label_encoder: Fitted LabelEncoder to decode predictions.
        device: Device to run inference on.
        output_file: Path to save the submission CSV.
    """
    model.eval()
    results = []

    print(f"Generating predictions for {len(test_loader.dataset)} images...")

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)

            # Forward pass with TTA (Horizontal Flip)
            outputs = model(images)
            outputs_flip = model(torch.flip(images, dims=[3]))
            outputs_tta = (outputs + outputs_flip) / 2.0

            # Get top 5 predictions
            _, preds = outputs_tta.topk(5, 1, True, True)
            preds = preds.cpu().numpy()

            for fname, pred_indices in zip(filenames, preds):
                # Decode labels (integers -> strings)
                pred_labels = label_encoder.inverse_transform(pred_indices)

                # Format as space-separated string
                pred_str = " ".join(pred_labels)

                results.append({"Image": fname, "Id": pred_str})

    # Create DataFrame and save
    df = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
