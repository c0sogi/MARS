import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library import config, utils, model, dataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = utils.AverageMeter()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).float().view(-1, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), inputs.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = utils.AverageMeter()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            losses.update(loss.item(), inputs.size(0))
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate AUC
    # Flatten arrays
    all_targets = np.array(all_targets).flatten()
    all_probs = np.array(all_probs).flatten()

    auc_score = utils.calculate_roc_auc(all_targets, all_probs)

    return losses.avg, auc_score


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print(f"Generating submission for {len(loader.dataset)} test clips...")
    model.eval()
    clips_list = []
    probs_list = []

    with torch.no_grad():
        for inputs, clips in loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            clips_list.extend(clips)
            probs_list.extend(probs.cpu().numpy().flatten())

    # Save submission
    utils.save_submission(clips_list, probs_list, output_path)
    print(f"Submission saved to {output_path}")


def run_training(epochs=config.EPOCHS, load_cached_data=True):
    """
    Main execution function for training and submission.
    """
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    print(f"Initializing model: {config.MODEL_NAME}")
    net = model.WhaleEfficientNet(pretrained=config.PRETRAINED)
    net = net.to(device)

    # 4. Optimization Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
    )

    # 5. Training Loop
    best_score = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(net, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | LR: {current_lr:.2e}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpointing & Early Stopping
        if val_auc > best_score:
            print(
                f"Validation AUC improved from {best_score} to {val_auc}. Saving model..."
            )
            best_score = val_auc
            utils.save_checkpoint(net, optimizer, epoch, best_score, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

        print("-" * 30)

    # 6. Inference
    print("\nTraining complete. Loading best model for inference...")

    # Re-initialize model structure to ensure clean state, then load weights
    best_net = model.WhaleEfficientNet(
        pretrained=False
    )  # Pretrained weights not needed as we load checkpoint
    best_net = best_net.to(device)

    _, loaded_score = utils.load_checkpoint(best_net, best_model_path, device=device)
    print(f"Loaded model with Validation AUC: {loaded_score}")

    generate_submission(best_net, test_loader, device, config.SUBMISSION_PATH)
