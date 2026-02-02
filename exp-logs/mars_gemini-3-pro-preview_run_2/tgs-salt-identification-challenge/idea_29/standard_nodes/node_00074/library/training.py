import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import inspect
from library.config import Config
from library.utils import rle_encode, unpad_image, calc_map
from library.losses import LovaszHingeLoss, StudentLoss


class Trainer:
    """
    Handles training and validation logic for both Teacher and Student models.
    """

    def __init__(
        self, model, device, optimizer=None, scheduler=None, teacher_model=None
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.teacher_model = teacher_model

        # Determine if model needs depth input by inspecting forward method
        sig = inspect.signature(model.forward)
        self.needs_depth = "depth" in sig.parameters

        # Loss functions
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()
        self.student_loss_fn = StudentLoss()

    def train_epoch(self, loader, is_student=False):
        """
        Runs one epoch of training.
        """
        self.model.train()
        if self.teacher_model:
            self.teacher_model.eval()

        total_loss = 0.0
        num_batches = len(loader)

        for batch in loader:
            images = batch["image"].to(self.device).float()
            depths = batch["depth"].to(self.device).float()
            masks = batch["mask"].to(self.device).float()  # (B, 1, H, W)

            self.optimizer.zero_grad()

            if is_student:
                # Student Forward: Input is Image only
                outputs = self.model(images)  # Returns dict {'logits', 'depth'}

                # Teacher Forward (Distillation)
                teacher_logits = None
                if self.teacher_model:
                    with torch.no_grad():
                        # Teacher needs depth
                        teacher_logits = self.teacher_model(images, depths)

                # Multi-task Loss
                loss, _ = self.student_loss_fn(outputs, masks, depths, teacher_logits)

            else:
                # Teacher Forward: Input is Image + Depth
                logits = self.model(images, depths)

                # Teacher Loss: Lovasz + BCE
                l_lovasz = self.lovasz(logits, masks)
                l_bce = self.bce(logits, masks)
                loss = l_lovasz + l_bce

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches

    def validate(self, loader):
        """
        Evaluates the model on the validation set.
        Performs threshold optimization to maximize mAP.
        """
        self.model.eval()
        all_preds = []
        all_masks = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device).float()
                depths = batch["depth"].to(self.device).float()
                masks = batch["mask"].numpy()  # Keep as numpy for post-processing

                # Forward pass
                if self.needs_depth:
                    logits = self.model(images, depths)
                else:
                    outputs = self.model(images)
                    logits = outputs["logits"]

                # Convert to probabilities
                probs = torch.sigmoid(logits).cpu().numpy()

                # Unpad predictions and masks to original size (101x101)
                for i in range(len(probs)):
                    # Squeeze channel dim: (1, 128, 128) -> (128, 128)
                    p = probs[i].squeeze(0)
                    m = masks[i].squeeze(0)

                    p_un = unpad_image(p)
                    m_un = unpad_image(m)

                    all_preds.append(p_un)
                    all_masks.append(m_un)

        all_preds = np.array(all_preds)
        all_masks = np.array(all_masks)

        # Threshold Optimization
        # We search for the best threshold that maximizes mAP
        best_map = 0.0
        best_thresh = 0.5

        # Search range: 0.30 to 0.70 with step 0.02
        thresholds = np.arange(0.3, 0.72, 0.02)

        for t in thresholds:
            # Binarize predictions
            binary_preds = (all_preds > t).astype(np.uint8)
            # Calculate mAP
            score = calc_map(binary_preds, all_masks)

            if score > best_map:
                best_map = score
                best_thresh = t

        return best_map, best_thresh


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    config,
    teacher_model=None,
    is_student=False,
):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    # Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    num_epochs = config.NUM_EPOCHS_STUDENT if is_student else config.NUM_EPOCHS_TEACHER
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    trainer = Trainer(model, device, optimizer, scheduler, teacher_model)

    best_map = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.CACHE_DIR, "best_model.pth")
    thresh_path = os.path.join(config.CACHE_DIR, "best_threshold.txt")

    print(f"Starting training (Student Mode: {is_student})...")

    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader, is_student)
        val_map, val_thresh = trainer.validate(val_loader)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.8f} | Val mAP: {val_map:.10f} | Best Thresh: {val_thresh:.4f}"
        )

        # Checkpointing
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # Save the best threshold for inference
            with open(thresh_path, "w") as f:
                f.write(str(val_thresh))
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model state before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    return model


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()

    # Load best threshold
    thresh_path = os.path.join(Config.CACHE_DIR, "best_threshold.txt")
    threshold = 0.5
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            try:
                threshold = float(f.read().strip())
            except ValueError:
                threshold = 0.5

    print(f"Generating submission with threshold: {threshold}")

    # Determine model input requirements
    sig = inspect.signature(model.forward)
    needs_depth = "depth" in sig.parameters

    submission_data = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device).float()
            depths = batch["depth"].to(device).float()
            ids = batch["id"]

            # Forward pass
            if needs_depth:
                logits = model(images, depths)
            else:
                outputs = model(images)
                logits = outputs["logits"]

            probs = torch.sigmoid(logits).cpu().numpy()

            for i, img_id in enumerate(ids):
                # Process each image in batch
                p = probs[i].squeeze(0)  # (128, 128)

                # Unpad to original size (101, 101)
                p_un = unpad_image(p)

                # Threshold and Encode
                mask = (p_un > threshold).astype(np.uint8)
                rle = rle_encode(mask)

                submission_data.append([img_id, rle])

    # Save to CSV
    df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
