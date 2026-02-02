import os
import cv2
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything
from library.data_setup import create_dataloaders
from library.model_setup import build_model
from library.engine import fit, evaluate, generate_submission


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set and correlates them with image features.
    """
    print("\nStarting Failure Analysis...")
    model.eval()

    all_probs = []
    all_labels = []

    # 1. Get Model Predictions
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 2. Calculate Error Magnitude
    # Error defined as (1.0 - probability assigned to the true class)
    # Higher value = worse prediction
    indices = np.arange(len(all_labels))
    true_class_probs = all_probs[indices, all_labels]
    errors = 1.0 - true_class_probs

    # 3. Extract Image Features
    # Access the underlying dataframe from the dataset
    val_df = val_loader.dataset.df

    features = []

    # Iterate through validation data to extract meta-features
    # Using the dataframe order which matches the loader (shuffle=False)
    for idx, row in val_df.iterrows():
        # Construct path (Config.INPUT_ROOT + relative path)
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        # Feature: File Size
        if os.path.exists(full_path):
            f_size = os.path.getsize(full_path)

            # Feature: Image Stats (Read image)
            img = cv2.imread(full_path)
            if img is not None:
                h, w, c = img.shape
                mean_intensity = img.mean()
                aspect_ratio = w / h if h > 0 else 0
            else:
                mean_intensity = 0
                aspect_ratio = 0
        else:
            f_size = 0
            mean_intensity = 0
            aspect_ratio = 0

        features.append(
            {
                "file_size": f_size,
                "mean_intensity": mean_intensity,
                "aspect_ratio": aspect_ratio,
                "error": errors[idx],
            }
        )

    feat_df = pd.DataFrame(features)

    # 4. Calculate Correlations
    print("Correlation between Error Magnitude and Input Features:")
    analysis_cols = ["file_size", "mean_intensity", "aspect_ratio"]

    for col in analysis_cols:
        if feat_df[col].std() > 0:  # Avoid constant columns
            corr, _ = pearsonr(feat_df[col], feat_df["error"])
            print(f"  {col}: {corr:.8f}")
        else:
            print(f"  {col}: N/A (Constant value)")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    # Using full dataset but limiting epochs for speed as per 'Fast Baseline' requirement
    train_loader, val_loader, test_loader = create_dataloaders(debug=False)

    # 3. Model Construction
    model = build_model(pretrained=True, num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10
    )  # Reduced epochs for speed

    # 5. Training
    print("Starting training...")
    # Using 10 epochs to ensure completion within strict time limits while maintaining performance
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=10,
        patience=5,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 6. Evaluation
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    loss, roc_auc = evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {roc_auc}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission
    print("\nGenerating submission...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
