import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from library import config
from library import utils
from library import dataset
from library import model as model_lib


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for specs, labels in loader:
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(specs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * specs.size(0)
        count += specs.size(0)

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            labels = labels.to(device)

            outputs = model(specs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * specs.size(0)

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            count += specs.size(0)

    avg_loss = running_loss / count if count > 0 else 0.0
    accuracy = correct / count if count > 0 else 0.0

    return avg_loss, accuracy


def run_training(
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    patience=5,
):
    """
    Main function to run the training pipeline.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for optimizer.
        patience (int): Early stopping patience.
    """
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader = dataset.get_dataloaders(
        batch_size=batch_size, num_workers=config.NUM_WORKERS
    )

    # 3. Model Initialization
    model = model_lib.ConvNeXtSpeech(
        num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED
    )
    model = model.to(device)

    print(f"Model initialized: {config.MODEL_NAME}")
    print(f"Trainable parameters: {utils.count_parameters(model)}")

    # 4. Optimizer, Loss, Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # Scheduler: Warmup -> Cosine Annealing
    # Note: Schedulers are stepped per epoch here
    warmup_epochs = config.WARMUP_EPOCHS
    # Ensure T_max is at least 1 to avoid ZeroDivisionError in CosineAnnealingLR
    # when total epochs <= warmup_epochs (e.g. in demo runs)
    main_epochs = max(1, epochs - warmup_epochs)

    scheduler1 = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    scheduler2 = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=config.MIN_LR)

    scheduler = SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs]
    )

    # 5. Training Loop
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Log metrics
        utils.log_metrics(epoch, train_loss, val_loss, val_acc, elapsed)

        # Checkpoint & Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            utils.save_checkpoint(
                state=model.state_dict(),
                is_best=True,
                filename=config.MODEL_CHECKPOINT_PATH,
            )
            epochs_no_improve = 0
            print(f"  -> New best model saved! Acc: {best_val_acc}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training completed. Best Validation Accuracy: {best_val_acc}")


def predict_and_submit(
    model_path=config.MODEL_CHECKPOINT_PATH,
    output_path=config.SUBMISSION_PATH,
    batch_size=config.BATCH_SIZE,
):
    """
    Loads the best model, runs inference on the test set, and generates the submission CSV.
    """
    device = torch.device(config.DEVICE)

    # 1. Load Model
    model = model_lib.ConvNeXtSpeech(
        num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED
    )
    model, best_acc = utils.load_checkpoint(
        model, filename=model_path, device=config.DEVICE
    )
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {model_path} (Best Val Acc: {best_acc})")

    # 2. Load Test Data
    test_loader = dataset.get_test_dataloader(
        batch_size=batch_size, num_workers=config.NUM_WORKERS
    )

    # 3. Inference
    all_fnames = []
    all_preds = []

    print("Starting inference on test set...")
    with torch.no_grad():
        for specs, _, fnames in test_loader:
            specs = specs.to(device)

            outputs = model(specs)
            # Get predicted class indices
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_fnames.extend(fnames)

    # 4. Generate Submission
    # Map indices back to labels
    pred_labels = [config.ID2LABEL[idx] for idx in all_preds]

    df_submission = pd.DataFrame({"fname": all_fnames, "label": pred_labels})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df_submission)} rows.")
