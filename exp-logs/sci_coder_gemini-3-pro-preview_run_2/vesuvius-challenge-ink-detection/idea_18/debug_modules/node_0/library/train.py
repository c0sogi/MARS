import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import InkDataset, TestInkDataset
from library.model import SegFormerB2
from library.utils import dice_loss, fbeta_score, EarlyStopping, rle_encoding, sigmoid


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    # BCEWithLogitsLoss combines Sigmoid and BCE
    bce_criterion = nn.BCEWithLogitsLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # Loss calculation: BCE + Dice
        bce = bce_criterion(outputs, labels)

        # Apply sigmoid for Dice calculation (since outputs are logits)
        pred_probs = torch.sigmoid(outputs)
        dice = dice_loss(pred_probs, labels)

        loss = bce + dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using the F0.5 score.
    """
    model.eval()
    running_score = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            pred_probs = torch.sigmoid(outputs)

            # Calculate F0.5 Score
            score = fbeta_score(
                pred_probs, labels, beta=Config.METRIC_BETA, threshold=Config.THRESHOLD
            )
            running_score += score.item()

    return running_score / len(dataloader)


def inference(model, device):
    """
    Generates submission.csv using Multi-View Ensemble Scanning and TTA.
    """
    print("Starting Inference with Multi-View Ensemble Scanning...")
    model.eval()

    # Read test metadata to get fragment IDs
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    fragment_ids = test_df["fragment_id"].unique()

    submission_data = []

    for fid in fragment_ids:
        # Instantiate a dummy dataset to get the original fragment dimensions
        ds_temp = TestInkDataset(fid, view="B")
        h_orig, w_orig = ds_temp.h, ds_temp.w

        # Accumulators for each view's full prediction map
        view_preds = []

        # Iterate through the 3 discrete views (A, B, C)
        for view in ["A", "B", "C"]:
            ds = TestInkDataset(fid, view=view)
            loader = DataLoader(
                ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            # Allocate memory for the full view prediction
            full_pred_view = torch.zeros(
                (h_orig, w_orig), dtype=torch.float32, device="cpu"
            )

            with torch.no_grad():
                for images, coords, sizes, _ in loader:
                    images = images.to(device)

                    # --- Test Time Augmentation (TTA) ---

                    # 1. Original
                    out = model(images)
                    preds = torch.sigmoid(out)

                    # 2. Horizontal Flip
                    images_flip = torch.flip(images, [3])
                    out_flip = model(images_flip)
                    preds_flip = torch.flip(torch.sigmoid(out_flip), [3])

                    # 3. Vertical Flip
                    images_vflip = torch.flip(images, [2])
                    out_vflip = model(images_vflip)
                    preds_vflip = torch.flip(torch.sigmoid(out_vflip), [2])

                    # Average predictions
                    batch_preds = (preds + preds_flip + preds_vflip) / 3.0

                    # Move to CPU for placement
                    batch_preds = batch_preds.cpu()

                    # Place patches into the full view map
                    for i in range(images.size(0)):
                        # coords is (B, 2) -> [x, y]
                        bx = coords[i, 0].item()
                        by = coords[i, 1].item()

                        # Prediction is (1, H, W), take first channel
                        patch = batch_preds[i, 0, :, :]

                        # Determine valid placement dimensions (handle right/bottom edges)
                        # The dataset pads the image if it's smaller than tile_size,
                        # so we must crop the prediction to the valid area.
                        valid_h = min(Config.TILE_SIZE, h_orig - by)
                        valid_w = min(Config.TILE_SIZE, w_orig - bx)

                        full_pred_view[by : by + valid_h, bx : bx + valid_w] = patch[
                            :valid_h, :valid_w
                        ]

            view_preds.append(full_pred_view)

        # --- Max-Fusion ---
        # Stack views: (3, H, W)
        stack = torch.stack(view_preds, dim=0)
        # Take the maximum probability across views per pixel
        final_prob, _ = torch.max(stack, dim=0)

        # Thresholding
        binary_mask = (final_prob > Config.THRESHOLD).numpy().astype(np.uint8)

        # Run-Length Encoding
        rle = rle_encoding(binary_mask)
        submission_data.append({"Id": fid, "Predicted": rle})

        # Explicit garbage collection to free large tensors
        del view_preds, stack, final_prob, binary_mask, full_pred_view
        gc.collect()

    # Save submission file
    sub_df = pd.DataFrame(submission_data)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_model():
    """
    Main training routine.
    """
    set_seed()
    device = torch.device(Config.DEVICE)

    # Initialize Datasets and Loaders
    train_ds = InkDataset(mode="train", limit=Config.MAX_TRAIN_SAMPLES)
    val_ds = InkDataset(mode="validation", limit=Config.MAX_VAL_SAMPLES)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = SegFormerB2()
    model.to(device)

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Early Stopping
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=True, path=checkpoint_path
    )

    print(
        f"Starting training for {Config.NUM_EPOCHS} epochs on device: {Config.DEVICE}"
    )

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss} - Val F0.5: {val_score}"
        )

        # Check Early Stopping (saves model if score improves)
        early_stopping(val_score, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model weights for inference
    print(f"Loading best model from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Generate Submission
    inference(model, device)
