import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_rmse,
)
from library.dataset import get_dataloaders
from library.model import PawpularitySwinModel


def extract_features(model, loader, device):
    """
    Extracts image embeddings and metadata features from the loader.
    Cite solution_lesson_node_00001: Decouple feature extraction from learning.
    """
    model.eval()
    embeddings = []
    meta_features = []
    targets = []
    ids = []

    with torch.no_grad():
        for batch_data in loader:
            images = batch_data["image"].to(device)
            features = batch_data["features"]
            batch_ids = batch_data["id"]

            # TTA: Original + Horizontal Flip
            # Extract image features using the backbone
            img_emb1 = model.backbone(images)
            img_emb2 = model.backbone(torch.flip(images, dims=[-1]))

            # Handle different output shapes (B, C) vs (B, L, C) vs (B, C, H, W)
            def pool_features(emb):
                if emb.ndim == 2:
                    return emb
                elif emb.ndim == 3:
                    return emb.mean(dim=1)  # (B, L, C) -> (B, C)
                elif emb.ndim == 4:
                    return emb.mean(dim=[-2, -1])  # (B, C, H, W) -> (B, C)
                return emb

            img_emb1 = pool_features(img_emb1)
            img_emb2 = pool_features(img_emb2)

            # Average embeddings
            img_emb = (img_emb1 + img_emb2) / 2.0

            embeddings.append(img_emb.cpu().numpy())
            meta_features.append(features.numpy())
            ids.extend(batch_ids)

            if "target" in batch_data:
                targets.append(batch_data["target"].numpy())

    embeddings = np.vstack(embeddings)
    meta_features = np.vstack(meta_features)

    if targets:
        targets = np.concatenate(targets)
        # Targets in loader are [0, 1], scale back to [0, 100] for Ridge
        targets = targets * 100.0
    else:
        targets = None

    return embeddings, meta_features, targets, ids


def run_failure_analysis(preds, targets, features, feature_names):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between absolute error and input features.
    """
    # Calculate residuals (absolute error)
    errors = np.abs(targets - preds)

    # Create a DataFrame for correlation analysis
    analysis_df = pd.DataFrame(features, columns=feature_names)
    analysis_df["Error"] = errors

    # Calculate correlation
    correlations = analysis_df.corr()["Error"].drop("Error")

    print("\n=== Failure Analysis: Correlation with Error Magnitude ===")
    print(correlations.sort_values(ascending=False).to_string())
    print("========================================================\n")


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = torch.device(Config.device)

    print(f"Device: {device}")
    print(f"Model: {Config.model_name}")
    print(
        "Strategy: Linear Probing with Ridge Regression (Cite solution_lesson_node_00005, solution_lesson_node_00006)"
    )

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 3. Model Initialization (Feature Extractor)
    print("Initializing backbone model...")
    model = PawpularitySwinModel(pretrained=True)
    model.to(device)

    # 4. Feature Extraction
    print("Extracting features...")
    X_train_img, X_train_meta, y_train, _ = extract_features(
        model, train_loader, device
    )
    X_val_img, X_val_meta, y_val, _ = extract_features(model, val_loader, device)

    # Concatenate features
    X_train = np.hstack([X_train_img, X_train_meta])
    X_val = np.hstack([X_val_img, X_val_meta])

    print(f"Feature shape: {X_train.shape}")

    # 5. Train Ridge Regression
    print("Training Ridge Regression with RidgeCV...")
    # Using StandardScaler to normalize features before Ridge (Cite solution_lesson_node_00007)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Use RidgeCV to automatically find the best alpha
    # Expanded alpha range to handle boundary clipping and higher dimensionality (Cite solution_lesson_node_00010)
    ridge = RidgeCV(alphas=[1.0, 10.0, 100.0, 500.0, 1000.0, 2500.0, 5000.0])
    ridge.fit(X_train_scaled, y_train)
    print(f"Best alpha: {ridge.alpha_}")

    # 6. Evaluation
    val_preds = ridge.predict(X_val_scaled)
    # Clip predictions to valid range [1, 100]
    val_preds = np.clip(val_preds, 1.0, 100.0)

    final_rmse = calculate_rmse(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_rmse}")

    # Failure Analysis
    run_failure_analysis(val_preds, y_val, X_val_meta, Config.feature_cols)

    # 7. Submission
    # Threshold from requirements: 17.735125135690733
    THRESHOLD = 17.735125135690733

    if final_rmse < THRESHOLD:
        print(
            f"Validation RMSE ({final_rmse}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test
        X_test_img, X_test_meta, _, test_ids = extract_features(
            model, test_loader, device
        )
        X_test = np.hstack([X_test_img, X_test_meta])
        X_test_scaled = scaler.transform(X_test)

        test_preds = ridge.predict(X_test_scaled)
        test_preds = np.clip(test_preds, 1.0, 100.0)

        submission_df = pd.DataFrame({"Id": test_ids, "Pawpularity": test_preds})
        submission_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"Validation RMSE ({final_rmse}) did not meet threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
