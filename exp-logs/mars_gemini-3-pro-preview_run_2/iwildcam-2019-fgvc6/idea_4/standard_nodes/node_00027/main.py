import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import f1_score
from library import config, dataset, model


def main():
    # =========================================================================
    # 1. Configuration for Fast Baseline
    # =========================================================================
    # Override config for speed while maintaining enough signal
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 30000  # ~20% of training data
    config.NUM_EPOCHS_STAGE1 = 2
    config.NUM_EPOCHS_STAGE2 = 2

    # Ensure reproducibility
    config.seed_everything(config.SEED)
    print(
        f"Starting execution with DEBUG={config.DEBUG}, SAMPLE_SIZE={config.DEBUG_SAMPLE_SIZE}"
    )

    # =========================================================================
    # 2. Data Loading (Training Subset)
    # =========================================================================
    train_loader, val_loader, _ = dataset.get_dataloaders()

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Initializing EfficientNet-B4 Native...")
    net = model.EfficientNetB4Native(
        num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED
    )
    net = net.to(config.DEVICE)

    # Loss setup
    if config.USE_CLASS_WEIGHTS:
        # Note: calculate_class_weights reads the full metadata file, so it's accurate even in debug
        class_weights = dataset.calculate_class_weights()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_model_wts = copy.deepcopy(net.state_dict())

    # =========================================================================
    # 4. Stage 1: Training Head (Frozen Backbone)
    # =========================================================================
    print("\n=== Stage 1: Training Head ===")
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
            f"Epoch {epoch+1}/{config.NUM_EPOCHS_STAGE1} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(net.state_dict())

    # Load best weights
    net.load_state_dict(best_model_wts)

    # =========================================================================
    # 5. Stage 2: Fine-Tuning (Unfreeze Top Blocks)
    # =========================================================================
    print("\n=== Stage 2: Fine-Tuning ===")
    net.unfreeze_blocks(n_blocks=3)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=config.LEARNING_RATE_STAGE2,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS_STAGE2
    )

    for epoch in range(config.NUM_EPOCHS_STAGE2):
        train_loss, train_f1 = model.train_one_epoch(
            net, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss, val_f1 = model.validate(net, val_loader, criterion, config.DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS_STAGE2} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(net.state_dict())

    # =========================================================================
    # 6. Full Validation Assessment
    # =========================================================================
    print("\n=== Running Full Validation Assessment ===")
    # Disable debug to load full validation set
    config.DEBUG = False
    # Re-initialize loaders to get full validation and test sets
    _, val_loader_full, test_loader_full = dataset.get_dataloaders()

    # Load best model
    net.load_state_dict(best_model_wts)

    # Calculate metric on full validation set
    val_loss, final_val_f1 = model.validate(
        net, val_loader_full, criterion, config.DEVICE
    )
    print(f"Final Validation Metric: {final_val_f1}")

    # =========================================================================
    # 7. Failure Analysis
    # =========================================================================
    print("\n=== Performing Failure Analysis ===")
    net.eval()
    all_preds = []
    all_targets = []

    # Collect predictions
    with torch.no_grad():
        for images, targets in val_loader_full:
            images = images.to(config.DEVICE)
            outputs = net(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Load metadata
    val_meta = pd.read_csv(config.VAL_METADATA_PATH)

    if len(val_meta) == len(all_preds):
        val_meta["pred"] = all_preds
        val_meta["target"] = all_targets
        val_meta["error"] = (val_meta["pred"] != val_meta["target"]).astype(int)

        # Feature Engineering for Analysis
        if "date_captured" in val_meta.columns:
            # Handle potential parsing errors gracefully
            val_meta["hour"] = pd.to_datetime(
                val_meta["date_captured"], errors="coerce"
            ).dt.hour
            val_meta["hour"] = val_meta["hour"].fillna(-1)

        features_to_check = ["location", "seq_num_frames", "frame_num", "hour"]
        correlations = {}

        print("Correlation between Error Magnitude and Input Features:")
        for feat in features_to_check:
            if feat in val_meta.columns:
                # Ensure numeric
                if pd.api.types.is_numeric_dtype(val_meta[feat]):
                    corr = val_meta["error"].corr(val_meta[feat])
                    correlations[feat] = corr
                    print(f"  {feat}: {corr:.6f}")
    else:
        print(
            "Warning: Metadata length does not match prediction length. Skipping detailed failure analysis."
        )

    # =========================================================================
    # 8. Submission Generation
    # =========================================================================
    threshold = 0.3978880094708815

    if final_val_f1 > threshold:
        print("\n=== Generating Submission ===")
        ids, preds = model.predict(net, test_loader_full, config.DEVICE)

        # Create submission directory
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"

        # Create DataFrame
        # Format: Id, Predicted
        submission_df = pd.DataFrame({"Id": ids, "Predicted": preds})
        submission_df.to_csv(submission_path, index=False)

        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation metric {final_val_f1} is below threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
