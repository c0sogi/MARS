import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import cv2

from library.config import Config
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterResNet
from library.engine import fit, predict, validate, set_seed


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    # Override Config for a fast baseline execution as per requirements
    Config.EPOCHS = 10

    # Create necessary output directories
    Config.create_output_dirs()

    # Set reproducibility
    set_seed(Config.SEED)

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create Datasets
    train_dataset = CatheterDataset(
        df_train, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = CatheterDataset(df_val, transforms=get_transforms("val"), mode="val")
    test_dataset = CatheterDataset(
        df_test, transforms=get_transforms("test"), mode="test"
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = CatheterResNet(pretrained=True, num_classes=Config.NUM_CLASSES)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------
    print("Starting training...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.CHECKPOINT_PATH,
    )

    # -------------------------------------------------------------------------
    # 5. Validation & Metrics
    # -------------------------------------------------------------------------
    print("Evaluating best model...")
    # Load the best checkpoint saved during training
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))

    avg_auc, class_aucs, val_loss = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {avg_auc}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    model.eval()

    # 6.1 Get raw predictions and labels for the validation set
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            # Use AMP for consistency with training/inference
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 6.2 Calculate Error Magnitude (Mean Absolute Error per sample across all classes)
    errors = np.abs(all_preds - all_labels).mean(axis=1)

    # 6.3 Extract Input Features (Width, Height, Aspect Ratio)
    # We iterate through the validation dataframe to read image dimensions.
    widths = []
    heights = []

    # Note: df_val aligns with val_loader because shuffle=False
    for _, row in df_val.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        # Read image to get dimensions
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
        else:
            # Fallback for missing images (though verification passed)
            widths.append(0)
            heights.append(0)

    widths = np.array(widths)
    heights = np.array(heights)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_ratios = np.where(heights > 0, widths / heights, 0)

    # 6.4 Calculate Correlations
    # Filter out any invalid images
    mask = widths > 0

    if mask.sum() > 1:
        corr_w = np.corrcoef(errors[mask], widths[mask])[0, 1]
        corr_h = np.corrcoef(errors[mask], heights[mask])[0, 1]
        corr_ar = np.corrcoef(errors[mask], aspect_ratios[mask])[0, 1]

        print(f"Correlation (Error vs Width): {corr_w:.4f}")
        print(f"Correlation (Error vs Height): {corr_h:.4f}")
        print(f"Correlation (Error vs Aspect Ratio): {corr_ar:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    threshold = 0.9297346737284826
    if avg_auc > threshold:
        print(f"\nValidation AUC ({avg_auc}) > {threshold}. Generating submission...")
        predict(
            model=model,
            dataloader=test_loader,
            df_test=df_test,
            device=device,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(f"\nValidation AUC ({avg_auc}) <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
