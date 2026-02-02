import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, model, dataset


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        vol = batch["volume"].to(device)
        label = batch["label"].to(device)
        batch_size = vol.size(0)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(vol)

        # Loss calculation
        loss = criterion(outputs, label)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, raw predictions (probabilities), and targets.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in loader:
            vol = batch["volume"].to(device)
            label = batch["label"].to(device)
            batch_size = vol.size(0)

            outputs = model(vol)
            loss = criterion(outputs, label)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for metric calculation
            probs = torch.sigmoid(outputs)

            preds_list.append(probs.cpu())
            targets_list.append(label.cpu())

    avg_loss = running_loss / dataset_size
    all_preds = torch.cat(preds_list)
    all_targets = torch.cat(targets_list)

    return avg_loss, all_preds, all_targets


def generate_submission(model, threshold, device, limit=None):
    """
    Generates the submission.csv file by:
    1. Predicting on test patches.
    2. Stitching patches back into full fragment masks.
    3. Applying threshold and RLE encoding.
    """
    print("Generating submission...")
    model.eval()

    # Load test metadata to determine full image dimensions
    test_df = pd.read_csv(config.TEST_METADATA)
    if limit is not None:
        test_df = test_df.iloc[:limit]

    # Create a map for quick lookup of patch dimensions (w, h) using sample_id
    # dataset.py returns x, y but not w, h directly in the batch
    patch_dims = test_df.set_index("sample_id")[["w", "h"]].to_dict("index")

    # Initialize canvases for each fragment
    # Find max extents
    fragment_sizes = {}
    for _, row in test_df.iterrows():
        fid = row["fragment_id"]
        max_x = row["x"] + row["w"]
        max_y = row["y"] + row["h"]

        if fid not in fragment_sizes:
            fragment_sizes[fid] = [0, 0]
        fragment_sizes[fid][0] = max(fragment_sizes[fid][0], max_x)
        fragment_sizes[fid][1] = max(fragment_sizes[fid][1], max_y)

    fragment_masks = {
        fid: np.zeros((h, w), dtype=np.float32)
        for fid, (w, h) in fragment_sizes.items()
    }

    # Get test loader
    _, _, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE, limit=limit
    )

    with torch.no_grad():
        for batch in test_loader:
            vol = batch["volume"].to(device)
            sample_ids = batch["sample_id"]
            fragment_ids = batch["fragment_id"]
            xs = batch["x"]
            ys = batch["y"]

            outputs = model(vol)
            probs = torch.sigmoid(outputs)
            probs_np = probs.cpu().numpy()

            for i, sample_id in enumerate(sample_ids):
                fid = fragment_ids[i]
                x = xs[i].item()
                y = ys[i].item()
                prob_map = probs_np[i, 0, :, :]  # (H, W)

                # Retrieve valid width and height (ignoring padding)
                if sample_id in patch_dims:
                    w = patch_dims[sample_id]["w"]
                    h = patch_dims[sample_id]["h"]

                    # Crop the prediction to the valid area
                    valid_pred = prob_map[:h, :w]

                    # Place in canvas
                    fragment_masks[fid][y : y + h, x : x + w] = valid_pred

    # Process masks and write submission
    submission_data = []
    for fid in sorted(fragment_masks.keys()):
        mask_prob = fragment_masks[fid]
        binary_mask = (mask_prob > threshold).astype(np.uint8)

        rle_str = utils.rle_encode(binary_mask)
        submission_data.append({"Id": fid, "Predicted": rle_str})

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")


def run_training(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, limit=None):
    """
    Main execution function.
    """
    # Setup
    config.setup_directories()
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # DataLoaders
    train_loader, val_loader, _ = dataset.get_dataloaders(
        batch_size=batch_size, limit=limit
    )

    # Model
    net = model.SFRPNet().to(device)

    # Optimizer & Loss
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    # Handle positive class weighting
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training Loop Variables
    best_score = -1.0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    patience = 5
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_preds, val_targets = validate(net, val_loader, criterion, device)

        # Metric (F0.5 with default threshold 0.5 for monitoring)
        val_score = utils.fbeta_score(val_preds, val_targets, beta=0.5, threshold=0.5)

        elapsed = time.time() - start_time

        # Print full precision metrics
        print(
            f"Epoch {epoch}/{epochs} | Time: {elapsed}s | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F0.5: {val_score}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_score:
            best_score = val_score
            utils.save_checkpoint(net, optimizer, epoch, val_score, best_model_path)
            patience_counter = 0
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    # Post-Training: Threshold Optimization
    print("Loading best model for threshold optimization...")
    checkpoint = utils.load_checkpoint(net, best_model_path)

    # Get predictions from best model
    _, val_preds, val_targets = validate(net, val_loader, criterion, device)

    # Optimize threshold
    best_threshold, best_opt_score = utils.optimize_threshold(
        val_preds, val_targets, beta=0.5
    )
    print(
        f"Optimization Complete. Best Threshold: {best_threshold}, Optimized Val F0.5: {best_opt_score}"
    )

    # Save threshold
    threshold_path = os.path.join(config.WORKING_DIR, "best_threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(str(best_threshold))

    # Generate Submission
    generate_submission(net, best_threshold, device, limit=limit)
