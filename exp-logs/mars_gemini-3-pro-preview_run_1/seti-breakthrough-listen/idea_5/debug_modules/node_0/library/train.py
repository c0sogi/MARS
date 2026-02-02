import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library import config, utils, data, model


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = utils.AverageMeter()

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), inputs.size(0))

    return loss_meter.avg


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = utils.AverageMeter()
    preds = []
    valid_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), inputs.size(0))

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy())
            valid_targets.extend(targets.cpu().numpy())

    preds = np.array(preds).flatten()
    valid_targets = np.array(valid_targets).flatten()

    # Calculate ROC AUC
    score = utils.get_score(valid_targets, preds)

    return loss_meter.avg, score


def train_model(debug=config.DEBUG):
    """
    Main training loop with Early Stopping.
    """
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    print(f"Initializing training on {device}...")

    # DataLoaders
    train_loader = data.get_train_dataloader(debug=debug)
    val_loader = data.get_val_dataloader(debug=debug)

    # Model Initialization
    net = model.TimeDistributedResNet50GN()
    net = net.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.PCT_START,
        div_factor=config.DIV_FACTOR,
        final_div_factor=config.FINAL_DIV_FACTOR,
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Training State
    best_score = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(
            net, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_score = validate(net, val_loader, criterion, device)

        print(f"Epoch {epoch}/{config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_score}")

        # Early Stopping Check
        if val_score > best_score + config.EARLY_STOPPING_MIN_DELTA:
            best_score = val_score
            patience_counter = 0
            utils.save_checkpoint(
                net, optimizer, scheduler, epoch, best_score, best_model_path
            )
            print(f"New best score! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_score}")
    return best_model_path


def generate_submission(model_path, debug=config.DEBUG):
    """
    Generates predictions for the test set and saves the submission file.
    """
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    print("Generating submission...")

    # Initialize Model
    net = model.TimeDistributedResNet50GN()
    net = net.to(device)

    # Load Best Weights
    try:
        epoch, score = utils.load_checkpoint(model_path, net, device=config.DEVICE)
        print(f"Loaded model from {model_path} (Epoch {epoch}, Score {score})")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # Test DataLoader
    test_loader = data.get_test_dataloader(debug=debug)

    # Inference
    net.eval()
    preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy().flatten())

    # Load Test Metadata to get IDs
    test_metadata = pd.read_csv(config.TEST_METADATA)

    # Handle Debug Subsampling for Metadata
    if debug:
        if len(test_metadata) > config.DEBUG_SAMPLE_SIZE:
            test_metadata = test_metadata.sample(
                n=config.DEBUG_SAMPLE_SIZE, random_state=config.SEED
            ).reset_index(drop=True)

    # Verify alignment
    if len(preds) != len(test_metadata):
        print(
            f"Warning: Prediction count ({len(preds)}) matches metadata count ({len(test_metadata)}) mismatch."
        )

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_metadata["id"], "target": preds})

    # Save Submission
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def run():
    """
    Orchestrates the training and submission process.
    """
    best_model_path = train_model()
    generate_submission(best_model_path)
