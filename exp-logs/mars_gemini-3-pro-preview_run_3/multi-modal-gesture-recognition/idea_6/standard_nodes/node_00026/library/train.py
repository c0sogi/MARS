import os
import time
import torch
import torch.optim as optim
import numpy as np
from library import config, utils, model, loss, data_loader


def train_one_epoch(net, loader, criterion, optimizer, device):
    """
    Executes one epoch of training.
    """
    net.train()
    losses = utils.AverageMeter()

    for batch_idx, (features, targets) in enumerate(loader):
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass
        # Returns list: [logits_stage1, logits_stage2, logits_stage3]
        outputs = net(features)

        # Compute Total Cascaded Loss
        total_loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), features.size(0))

    return losses.avg


def validate(net, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Levenshtein Error Rate.
    """
    net.eval()
    losses = utils.AverageMeter()

    predictions = {}
    ground_truth = {}

    with torch.no_grad():
        for batch_data in loader:
            # Val loader returns (features, labels, sample_id)
            # Batch size is expected to be 1 for variable length sequences
            if len(batch_data) == 3:
                features, targets, sample_ids = batch_data
            else:
                features, targets = batch_data
                sample_ids = None

            features = features.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = net(features)

            # Compute Loss (for tracking purposes)
            batch_loss = criterion(outputs, targets)
            losses.update(batch_loss.item(), features.size(0))

            # Decode Predictions from Stage 3 (Final Refinement)
            logits_stage3 = outputs[2]
            probs = torch.softmax(logits_stage3, dim=1)

            for i in range(features.size(0)):
                # Permute to (Time, Classes) for decoding
                sample_probs = probs[i].permute(1, 0)

                # Decode predicted sequence
                pred_seq = utils.decode_predictions(sample_probs, threshold=5)

                # Reconstruct Ground Truth Sequence from frame-wise labels
                # Perform RLE on the target tensor to extract gesture IDs
                gt_frame_labels = targets[i].cpu().numpy()
                gt_seq = []
                if len(gt_frame_labels) > 0:
                    # Find indices where value changes
                    locs = np.where(gt_frame_labels[:-1] != gt_frame_labels[1:])[0] + 1
                    splits = np.split(gt_frame_labels, locs)
                    for seg in splits:
                        lbl = seg[0]
                        if lbl != 0:  # Ignore background
                            gt_seq.append(int(lbl))

                # Store for metric computation
                sid = sample_ids[i] if sample_ids else f"val_{i}"
                predictions[sid] = pred_seq
                ground_truth[sid] = gt_seq

    # Compute Levenshtein Error Rate
    score = utils.compute_dataset_score(predictions, ground_truth)

    return losses.avg, score


def generate_submission(net, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    net.eval()
    predictions = {}

    print(f"Generating predictions for {len(loader.dataset)} test samples...")

    with torch.no_grad():
        for batch_data in loader:
            # Test loader returns (features, labels, sample_id)
            features, _, sample_ids = batch_data
            features = features.to(device)

            # Forward
            outputs = net(features)

            # Use Stage 3 output
            logits_stage3 = outputs[2]
            probs = torch.softmax(logits_stage3, dim=1)

            for i in range(features.size(0)):
                sample_probs = probs[i].permute(1, 0)
                pred_seq = utils.decode_predictions(sample_probs, threshold=5)
                predictions[sample_ids[i]] = pred_seq

    utils.save_submission(predictions, output_path)


def run_training(num_epochs=config.NUM_EPOCHS, load_cached_data=True):
    """
    Main driver function for training and evaluation.
    """
    utils.set_seed()
    device = config.get_device()

    # 1. Load Data
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    print("Initializing Iterative Cascaded Network...")
    net = model.IterativeCascadedNet().to(device)

    # 3. Loss and Optimizer
    criterion = loss.CascadedLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_score = validate(net, val_loader, criterion, device)

        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Time: {duration:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Error Rate: {val_score:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved! (Score: {best_score:.10f})")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{config.PATIENCE}"
            )

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Final Inference
    print("Loading best model for submission generation...")
    if os.path.exists(config.MODEL_SAVE_PATH):
        net.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    else:
        print("Warning: No model file found. Using current weights.")

    generate_submission(net, test_loader, device, config.SUBMISSION_FILE)
