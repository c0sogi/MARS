import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from library import config, dataset, model


def run():
    """
    Executes the full training pipeline:
    1. Setup data and model.
    2. Stage 1: Train head with frozen backbone.
    3. Stage 2: Fine-tune top blocks with Early Stopping.
    4. Inference: Generate and save submission.
    """
    # 1. Setup
    config.seed_everything(config.SEED)
    print(f"Initializing training for {config.PROJECT_NAME}...")

    # Load Data
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    # Initialize Model
    net = model.EfficientNetB4Native(
        num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED
    )
    net = net.to(config.DEVICE)

    # Loss Function (Weighted for Class Imbalance)
    if config.USE_CLASS_WEIGHTS:
        class_weights = dataset.calculate_class_weights()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss()

    # Trackers for Best Model
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(net.state_dict())

    # ====================================================
    # Stage 1: Training Head (Backbone Frozen)
    # ====================================================
    print("\n=== Stage 1: Training Head (Backbone Frozen) ===")
    net.freeze_backbone()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=config.LEARNING_RATE_STAGE1,
    )

    for epoch in range(config.NUM_EPOCHS_STAGE1):
        train_loss, train_f1 = model.train_one_epoch(
            net, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss, val_f1 = model.validate(net, val_loader, criterion, config.DEVICE)

        print(
            f"Stage 1 - Epoch {epoch+1}/{config.NUM_EPOCHS_STAGE1} | "
            f"Train Loss: {train_loss:.6f} F1: {train_f1:.6f} | "
            f"Val Loss: {val_loss:.6f} F1: {val_f1:.10f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(net.state_dict())

    # Load best weights from Stage 1 before starting Stage 2
    net.load_state_dict(best_model_wts)

    # ====================================================
    # Stage 2: Fine-Tuning (Unfreezing Top Blocks)
    # ====================================================
    print("\n=== Stage 2: Fine-Tuning (Unfreezing Top Blocks) ===")
    net.unfreeze_blocks(n_blocks=3)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=config.LEARNING_RATE_STAGE2,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS_STAGE2
    )

    # Early Stopping Parameters
    patience = 5
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS_STAGE2):
        train_loss, train_f1 = model.train_one_epoch(
            net, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss, val_f1 = model.validate(net, val_loader, criterion, config.DEVICE)

        scheduler.step()

        print(
            f"Stage 2 - Epoch {epoch+1}/{config.NUM_EPOCHS_STAGE2} | "
            f"Train Loss: {train_loss:.6f} F1: {train_f1:.6f} | "
            f"Val Loss: {val_loss:.6f} F1: {val_f1:.10f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(net.state_dict())
            # Save checkpoint immediately
            torch.save(net.state_dict(), config.BEST_MODEL_PATH)
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nTraining Complete. Best Validation F1: {best_f1:.10f}")

    # ====================================================
    # Inference and Submission
    # ====================================================
    print("Generating submission...")

    # Load best model
    net.load_state_dict(best_model_wts)

    # Predict
    test_ids, test_preds = model.predict(net, test_loader, config.DEVICE)

    # Save
    submission_df = pd.DataFrame({"Id": test_ids, "Predicted": test_preds})
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
