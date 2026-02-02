import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import StegoDataset, get_transforms, load_metadata
from library.model import HPF_EfficientNet
from library.engine import fit, predict, valid_one_epoch


def perform_failure_analysis(model, loader, df_val, device):
    """
    Analyzes model errors on the validation set.
    Calculates correlation between Error Magnitude and File Size.
    """
    print("\n=== Performing Failure Analysis ===")
    model.eval()

    all_preds = []
    all_labels = []

    # 1. Collect Predictions
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy().ravel()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy().ravel())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 2. Calculate Error Magnitude
    # Error = |True Label - Predicted Probability|
    errors = np.abs(all_labels - all_preds)

    # 3. Extract Input Features (File Size)
    # We iterate through the dataframe to get file sizes corresponding to the images
    print("Extracting metadata features for correlation analysis...")
    file_sizes = []
    root = Config.input_root

    # Ensure the order matches the loader (sequential because shuffle=False)
    # The loader was created from df_val, so the order is preserved.
    for _, row in df_val.iterrows():
        full_path = os.path.join(root, row["file_path"])
        try:
            size = os.path.getsize(full_path)
        except OSError:
            size = 0
        file_sizes.append(size)

    file_sizes = np.array(file_sizes)

    # 4. Compute Correlation
    # Create a temporary DataFrame for correlation calculation
    analysis_df = pd.DataFrame(
        {"error": errors, "file_size": file_sizes, "label": all_labels}
    )

    corr_matrix = analysis_df.corr()
    error_filesize_corr = corr_matrix.loc["error", "file_size"]

    print("-" * 30)
    print(
        f"Correlation between Error Magnitude and File Size: {error_filesize_corr:.6f}"
    )
    print("-" * 30)

    # Check correlation with label (just for insight)
    error_label_corr = corr_matrix.loc["error", "label"]
    print(f"Correlation between Error Magnitude and Label: {error_label_corr:.6f}")
    print("Failure analysis complete.\n")


def main():
    # 1. Setup & Configuration
    Config.setup()
    device = Config.device
    seed_everything(Config.seed)

    print(f"Initializing Run. Device: {device}")
    print(f"Training for {Config.epochs} epochs.")

    # 2. Data Loading
    print("Loading metadata...")
    df_train = load_metadata(Config.train_csv)
    df_val = load_metadata(Config.val_csv)

    # Calculate class weights for imbalance handling
    # Dataset has 3 Stego (1) for every 1 Cover (0).
    # We want to balance the loss contribution.
    # pos_weight = num_neg / num_pos = 1/3 approx 0.333
    # Cite solution_lesson_node_00001: Addressing bias towards majority class observed in failure analysis.
    num_pos = df_train[df_train["label"] == 1].shape[0]
    num_neg = df_train[df_train["label"] == 0].shape[0]
    pos_weight_val = num_neg / num_pos
    print(
        f"Class Imbalance: Neg={num_neg}, Pos={num_pos}. Using pos_weight={pos_weight_val:.4f}"
    )
    pos_weight = torch.tensor([pos_weight_val]).to(device)

    # Initialize Datasets
    train_dataset = StegoDataset(df_train, transform=get_transforms(mode="train"))
    val_dataset = StegoDataset(df_val, transform=get_transforms(mode="val"))

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing HPF-EfficientNet...")
    model = HPF_EfficientNet().to(device)

    # 4. Optimizer, Loss, Scheduler
    # Using pos_weight to penalize False Positives on the minority Cover class
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.max_lr,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # 5. Training Loop
    best_model_path = os.path.join(Config.working_dir, "best_model.pth")

    print("Starting training loop...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.epochs,
        patience=Config.patience,
        save_path=best_model_path,
    )

    # 6. Final Validation & Metrics
    print("Loading best model for final validation...")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Training failed to produce a best_model.pth file.")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Compute metric on full validation set
    val_loss, val_score = valid_one_epoch(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, df_val, device)

    # 8. Submission
    target_threshold = 0.8244512275267252
    if val_score > target_threshold:
        print(
            f"Validation score {val_score} exceeds threshold {target_threshold}. Generating submission..."
        )
        predict(best_model_path, device, debug=False)
    else:
        print(
            f"Validation score {val_score} does not exceed threshold {target_threshold}. Skipping submission."
        )

    print("Process completed successfully.")


if __name__ == "__main__":
    main()
