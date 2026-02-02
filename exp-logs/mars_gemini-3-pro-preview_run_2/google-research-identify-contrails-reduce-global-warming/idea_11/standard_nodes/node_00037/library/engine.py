import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import rle_encode


class EarlyStopping:
    """
    Early stopping to stop the training when the validation score does not improve
    after a certain number of epochs.
    """

    def __init__(self, patience=5, delta=0, mode="max", save_path="checkpoint.pth"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        else:
            if self.mode == "max":
                if score < self.best_score + self.delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(score, model)
                    self.counter = 0
            elif self.mode == "min":
                if score > self.best_score - self.delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(score, model)
                    self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation score improves."""
        torch.save(model.state_dict(), self.save_path)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device, threshold=0.5, use_tta=False):
    """
    Evaluates the model on the validation set using Global Dice score.
    Applies Test-Time Augmentation (TTA) if enabled.
    """
    model.eval()

    # Accumulators for Global Dice calculation
    intersection_sum = 0.0
    cardinality_sum = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            if use_tta:
                # 1. Original
                pred_1 = torch.sigmoid(model(images))

                # 2. Horizontal Flip
                images_h = torch.flip(images, [3])
                pred_h = torch.sigmoid(model(images_h))
                pred_2 = torch.flip(pred_h, [3])

                # 3. Vertical Flip
                images_v = torch.flip(images, [2])
                pred_v = torch.sigmoid(model(images_v))
                pred_3 = torch.flip(pred_v, [2])

                # 4. Rotate 180 (Horizontal + Vertical Flip)
                images_hv = torch.flip(images, [2, 3])
                pred_hv = torch.sigmoid(model(images_hv))
                pred_4 = torch.flip(pred_hv, [2, 3])

                # Average predictions
                probs = (pred_1 + pred_2 + pred_3 + pred_4) / 4.0
            else:
                probs = torch.sigmoid(model(images))

            # Binarize predictions
            preds_bin = (probs > threshold).float()

            # Flatten tensors for global accumulation
            preds_flat = preds_bin.view(-1)
            masks_flat = masks.view(-1)

            intersection_sum += (preds_flat * masks_flat).sum().item()
            cardinality_sum += preds_flat.sum().item() + masks_flat.sum().item()

    # Compute Global Dice
    if cardinality_sum == 0:
        # If both sets are empty, Dice is 1.0. If only one is empty, it's 0.0.
        # Since cardinality = sum(pred) + sum(target), cardinality=0 implies both are empty.
        return 1.0

    dice_score = (2.0 * intersection_sum) / cardinality_sum
    return dice_score


def predict_and_submit(
    model,
    dataloader,
    device,
    threshold=0.5,
    use_tta=False,
    output_path=Config.SUBMISSION_PATH,
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    submission_data = []

    print("Generating predictions...")

    with torch.no_grad():
        for images, record_ids in dataloader:
            images = images.to(device)

            if use_tta:
                # 1. Original
                pred_1 = torch.sigmoid(model(images))

                # 2. Horizontal Flip
                images_h = torch.flip(images, [3])
                pred_h = torch.sigmoid(model(images_h))
                pred_2 = torch.flip(pred_h, [3])

                # 3. Vertical Flip
                images_v = torch.flip(images, [2])
                pred_v = torch.sigmoid(model(images_v))
                pred_3 = torch.flip(pred_v, [2])

                # 4. Rotate 180
                images_hv = torch.flip(images, [2, 3])
                pred_hv = torch.sigmoid(model(images_hv))
                pred_4 = torch.flip(pred_hv, [2, 3])

                probs = (pred_1 + pred_2 + pred_3 + pred_4) / 4.0
            else:
                probs = torch.sigmoid(model(images))

            # Binarize
            preds_bin = (probs > threshold).cpu().numpy().astype(np.uint8)

            # Iterate over batch to encode
            for i in range(images.size(0)):
                rid = record_ids[i]
                # Extract single channel mask: (1, H, W) -> (H, W)
                mask = preds_bin[i, 0, :, :]

                if mask.sum() == 0:
                    encoded_pixels = "-"
                else:
                    encoded_pixels = rle_encode(mask)

                submission_data.append(
                    {"record_id": rid, "encoded_pixels": encoded_pixels}
                )

    # Create DataFrame and save
    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    loss_fn,
    num_epochs=Config.EPOCHS,
):
    """
    Orchestrates the training process.
    """
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    early_stopping = EarlyStopping(patience=5, mode="max", save_path=best_model_path)

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn
        )

        # Step the scheduler (CosineAnnealing is typically stepped once per epoch)
        if scheduler is not None:
            scheduler.step()

        val_dice = validate(
            model,
            val_loader,
            device,
            threshold=Config.THRESHOLD,
            use_tta=Config.USE_TTA,
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss} | Val Global Dice: {val_dice}"
        )

        # Check early stopping
        early_stopping(val_dice, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training complete.")

    # Load best model weights for returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")

    return model
