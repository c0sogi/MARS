import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
from scipy.stats import pearsonr

# Import provided library modules
from library import config
from library import dataset
from library import model
from library import engine


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes the correlation between model error and input features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    all_probs = []
    all_targets = []

    # 1. Get Predictions and Targets
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            # Use AMP for inference speed
            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(images)
                probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 2. Calculate Error Magnitude (Mean Absolute Error per sample)
    # This represents how "confused" or wrong the model was
    errors = np.mean(np.abs(all_probs - all_targets), axis=1)

    # 3. Extract Input Features (Metadata)
    # We access the underlying dataframe from the dataset
    val_df = val_loader.dataset.df

    file_sizes = []
    widths = []
    heights = []
    aspect_ratios = []

    # Iterate through the dataframe to get file stats
    # Note: val_loader is not shuffled, so indices align
    for _, row in val_df.iterrows():
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])

        try:
            # File Size
            size = os.path.getsize(file_path)

            # Dimensions (Read header only if possible, but cv2 reads full)
            # For speed, we assume standard reading.
            # Given the constraints, we'll read the image to get accurate dims.
            img = cv2.imread(file_path)
            if img is not None:
                h, w, _ = img.shape
            else:
                h, w = 0, 0

            file_sizes.append(size)
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)

        except Exception as e:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # 4. Calculate Correlations
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "file_size": file_sizes,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
        }
    )

    features = ["file_size", "width", "height", "aspect_ratio"]
    print("Correlation between Error Magnitude and Input Features:")

    for feature in features:
        # Handle cases with zero variance or NaNs
        if analysis_df[feature].std() == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(analysis_df["error"], analysis_df[feature])
        print(f"  {feature}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # Using default batch sizes from config
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    # 3. Model Initialization
    print(f"Initializing model: {config.MODEL_NAME}")
    net = model.AppleDiseaseModel(pretrained=True)
    net.to(device)

    # 4. Training Configuration
    # Increased epochs for larger model convergence
    EPOCHS = 15
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    trained_model, best_f1 = engine.train_model(
        net,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=EPOCHS,
        patience=6,
    )

    # 6. Final Validation Metric
    # Re-calculate on the full validation set to ensure accuracy and correct formatting
    criterion = nn.BCEWithLogitsLoss()
    _, final_f1 = engine.evaluate(trained_model, val_loader, device, criterion)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_f1}")

    # 7. Failure Analysis
    perform_failure_analysis(trained_model, val_loader, device)

    # 8. Submission Generation
    # Only submit if we beat the previous baseline
    PREV_BEST_F1 = 0.9153778856820253
    if final_f1 > PREV_BEST_F1:
        print(
            f"Validation F1 ({final_f1:.4f}) improved over baseline ({PREV_BEST_F1:.4f}). Generating submission..."
        )
        engine.generate_submission(trained_model, test_loader, device)
    else:
        print(
            f"Validation F1 ({final_f1:.4f}) did not improve over baseline ({PREV_BEST_F1:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
