import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import cv2
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, load_checkpoint, quadratic_weighted_kappa
from library.dataset import create_dataloaders
from library.model import OrdinalEfficientNet
from library.engine import train_model, validate, predict


def get_validation_predictions(model, loader, device):
    """
    Runs inference on the validation set to retrieve raw predictions and targets
    for failure analysis.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            # targets are ordinal vectors (B, 4)

            outputs = model(images)

            # Decode predictions: sum probabilities and round
            pred_scores = outputs.sum(dim=1)
            pred_labels = pred_scores.round().cpu().numpy().astype(int)

            # Decode targets: sum binary vector
            target_labels = targets.sum(dim=1).cpu().numpy().astype(int)

            all_preds.extend(pred_labels)
            all_targets.extend(target_labels)

    return np.array(all_preds), np.array(all_targets)


def perform_failure_analysis(val_df, preds, targets):
    """
    Analyzes the correlation between model error and input image features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Error Magnitude (Absolute difference)
    errors = np.abs(preds - targets)
    val_df["error"] = errors

    # Feature Extraction
    # We extract basic image properties to see if image quality/size correlates with error
    widths = []
    heights = []
    aspect_ratios = []
    mean_intensities = []
    file_sizes = []

    print("Extracting features from validation images for analysis...")

    for idx, row in val_df.iterrows():
        # Construct full path
        file_path = os.path.join(Config.input_dir, row["file_path"])

        # File Size
        try:
            f_size = os.path.getsize(file_path)
        except OSError:
            f_size = 0
        file_sizes.append(f_size)

        # Image Stats
        # Reading image to get dimensions and intensity
        img = cv2.imread(file_path)
        if img is not None:
            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            mean_intensities.append(img.mean())
        else:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            mean_intensities.append(0)

    val_df["width"] = widths
    val_df["height"] = heights
    val_df["aspect_ratio"] = aspect_ratios
    val_df["mean_intensity"] = mean_intensities
    val_df["file_size"] = file_sizes

    # Calculate Correlations
    features = ["width", "height", "aspect_ratio", "mean_intensity", "file_size"]
    correlations = {}

    for feat in features:
        # Check for non-zero variance to avoid division by zero in correlation
        if val_df[feat].std() > 1e-6:
            # [0, 1] is the correlation between the two variables
            corr = np.corrcoef(val_df[feat], val_df["error"])[0, 1]
            correlations[feat] = corr
        else:
            correlations[feat] = 0.0

    print("\nCorrelation between Error Magnitude and Input Features:")
    for feat, corr in correlations.items():
        print(f"{feat}: {corr:.4f}")


def main():
    # 1. Setup Environment
    seed_everything(Config.seed)
    device = torch.device(Config.device)
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders()

    # 3. Model Configuration
    print(f"Initializing Model: {Config.backbone}...")
    model = OrdinalEfficientNet(
        backbone_name=Config.backbone, pretrained=Config.pretrained
    )
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.scheduler_eta_min
    )

    # 4. Training Loop
    print("Starting Training...")
    best_score = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=Config.epochs,
        patience=Config.early_stopping_patience,
    )

    # 5. Final Validation
    print("\nLoading best model for final evaluation...")
    load_checkpoint(model, filename="best_model.pth")

    val_loss, final_qwk = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_qwk}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis...")
    preds, targets = get_validation_predictions(model, val_loader, device)

    # Load validation metadata to associate predictions with image files
    val_df = pd.read_csv(Config.val_csv_path)

    if len(val_df) == len(preds):
        perform_failure_analysis(val_df, preds, targets)
    else:
        print(
            f"Warning: Mismatch in validation set size (DF: {len(val_df)}, Preds: {len(preds)}). Skipping analysis."
        )

    # 7. Submission Generation
    SUBMISSION_THRESHOLD = 0.9194950903896975

    if final_qwk > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_qwk}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Generate predictions for test set
        df_submission = predict(model, test_loader, device)

        # Save to ./submission/submission.csv
        output_dir = "./submission"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "submission.csv")

        df_submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    else:
        print(
            f"\nMetric ({final_qwk}) <= Threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
