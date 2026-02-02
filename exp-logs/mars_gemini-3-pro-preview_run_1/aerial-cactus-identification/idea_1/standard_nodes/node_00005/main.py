import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2

# Import provided library components
from library.utils import Config, set_seed
from library.dataset import get_dataloaders
from library.model import CactusResNet
from library.engine import train_one_epoch, evaluate, predict


def analyze_failures(model, val_loader, config):
    """
    Performs failure analysis on the validation set by correlating
    error magnitude with image meta-features.
    """
    print("SECTION: FAILURE ANALYSIS")

    # 1. Get Predictions and Targets
    model.eval()
    all_preds = []
    all_targets = []

    # Disable gradients for inference to optimize speed and memory
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(config.DEVICE)
            # Forward pass
            outputs = model(inputs)
            # Apply sigmoid to get probabilities (logits -> probs)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    # Calculate Error Magnitude (Absolute Error)
    errors = np.abs(targets - preds)

    # 2. Extract Image Meta-Features
    # Load validation metadata to get file paths
    # Note: val_loader is created with shuffle=False, so the order matches the metadata CSV
    val_df = pd.read_csv(config.VAL_METADATA_PATH)

    # Ensure alignment between predictions and metadata
    if len(val_df) != len(errors):
        print(
            f"Warning: Mismatch between validation set size ({len(val_df)}) and predictions ({len(errors)})."
        )
        return

    # Feature lists
    img_means = []
    img_stds = []
    file_sizes = []

    # Iterate through validation images to extract features
    # We use the raw images from disk to get original statistics
    for idx, row in val_df.iterrows():
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])

        # File size
        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0
        file_sizes.append(size)

        # Image stats (Intensity and Contrast)
        img = cv2.imread(file_path)
        if img is not None:
            # Calculate mean and std of pixel values
            img_means.append(img.mean())
            img_stds.append(img.std())
        else:
            img_means.append(0)
            img_stds.append(0)

    # 3. Calculate Correlations
    # Create a DataFrame for easy correlation calculation
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "mean_intensity": img_means,
            "contrast": img_stds,
            "file_size": file_sizes,
        }
    )

    print("Correlation between Error Magnitude and Input Features:")
    # Calculate Pearson correlation
    for feat in ["mean_intensity", "contrast", "file_size"]:
        corr = df_analysis["error"].corr(df_analysis[feat])
        print(f"  {feat}: Correlation = {corr:.16f}")


def main():
    # 1. Configuration
    # Initialize Config with specific epochs
    # Increased to 30 epochs to accommodate Mixup convergence (Cite solution_lesson_node_00004)
    config = Config(epochs=30, batch_size=128)

    # Set random seeds for reproducibility
    set_seed(config.SEED)

    # 2. Data Loading
    # Get DataLoaders for train, validation, and test sets
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # 3. Model Initialization
    # Initialize the custom ResNet model
    model = CactusResNet(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    # 4. Training Setup
    # Use BCEWithLogitsLoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    # AdamW optimizer for better weight decay handling
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2
    )
    # Cosine Annealing scheduler for smooth learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(config.CACHE_DIR, "best_model.pth")

    for epoch in range(config.NUM_EPOCHS):
        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )

        # Evaluate on validation set
        val_loss, val_auc = evaluate(model, val_loader, criterion, config.DEVICE)

        # Update learning rate
        scheduler.step()

        # Save best model based on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation
    # Load the best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))

    # Compute final validation metric
    _, final_auc = evaluate(model, val_loader, criterion, config.DEVICE)
    # Print exactly as required with full precision
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, config)

    # 8. Submission
    # Generate predictions for the test set and save to CSV
    # Only submit if we beat the previous best score
    PREV_BEST_SCORE = 0.9999980097310881

    if final_auc > PREV_BEST_SCORE:
        print(
            f"Validation metric {final_auc:.15f} > {PREV_BEST_SCORE:.15f}. Generating submission..."
        )
        predict(
            model=model,
            dataloader=test_loader,
            device=config.DEVICE,
            output_path=config.SUBMISSION_PATH,
            metadata_path=config.TEST_METADATA_PATH,
            debug_size=config.DEBUG_SUBSET_SIZE,
        )
    else:
        print(
            f"Validation metric {final_auc:.15f} did not beat threshold {PREV_BEST_SCORE:.15f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
