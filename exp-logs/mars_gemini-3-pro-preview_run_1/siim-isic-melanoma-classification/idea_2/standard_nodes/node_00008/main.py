import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

from library.config import (
    SEED,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_CHECKPOINT_PATH,
    SUBMISSION_PATH,
    TTA_STEPS,
    NUM_WORKERS,
    WORKING_DIR,
)
from library.utils import seed_everything, calculate_metric
from library.data_loader import get_dataloaders, preprocess_metadata
from library.model import EfficientNetFusion
from library.engine import train_model, predict_tta, generate_submission


def main():
    # 1. Setup Environment
    seed_everything(SEED)
    print(f"Using device: {DEVICE}")

    # 2. Load Data
    # Use cached data for speed
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Retrieve metadata and tabular arrays for analysis and TTA
    _, val_df, test_df, _, val_tab, test_tab = preprocess_metadata(
        load_cached_data=True
    )

    # Determine input dimensions
    num_tabular_features = train_loader.dataset.tabular_data.shape[1]
    print(f"Number of tabular features: {num_tabular_features}")

    # 3. Initialize Model
    model = EfficientNetFusion(num_tabular_features=num_tabular_features)
    model.to(DEVICE)

    # 4. Training Configuration
    # Increased epochs for OneCycleLR convergence (Cite solution_lesson_node_00007)
    NUM_EPOCHS = 10
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # Use OneCycleLR for better stability with larger backbone
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
    )

    # 5. Run Training
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        patience=NUM_EPOCHS,  # No early stopping needed for short run
        save_path=MODEL_CHECKPOINT_PATH,
        scheduler=scheduler,
    )

    # 6. Evaluation & Failure Analysis
    print("\nRunning Evaluation on Best Model...")
    # Reload best model
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    val_preds = []
    val_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for images, tabular, targets in val_loader:
            images = images.to(DEVICE)
            tabular = tabular.to(DEVICE)

            logits = model(images, tabular)
            probs = torch.sigmoid(logits)

            val_preds.extend(probs.cpu().numpy().flatten())
            val_targets.extend(targets.numpy().flatten())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Metric
    final_metric = calculate_metric(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation with Input Features
    print("\n=== Failure Analysis ===")
    errors = np.abs(val_targets - val_preds)

    # Correlate with Age (Feature 0 in processed tabular data)
    age_feature = val_tab[:, 0]
    if np.std(age_feature) > 0:
        corr_age, _ = pearsonr(errors, age_feature)
        print(f"Correlation (Error vs Age): {corr_age}")
    else:
        print("Correlation (Error vs Age): Undefined (Constant feature)")

    # Correlate with other tabular features (One-Hot Encoded)
    # We report the maximum correlation found to identify potential bias
    feature_corrs = []
    for i in range(1, val_tab.shape[1]):
        feat = val_tab[:, i]
        if np.std(feat) > 0:
            c, _ = pearsonr(errors, feat)
            feature_corrs.append(c)

    if feature_corrs:
        max_corr = max(np.abs(feature_corrs))
        print(f"Max Correlation (Error vs Categorical Features): {max_corr}")
    else:
        print("No valid correlations found for categorical features.")

    # 7. Submission Logic
    THRESHOLD = 0.874794288335701

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        predictions = predict_tta(
            model=model,
            test_df=test_df,
            test_tab=test_tab,
            tta_steps=TTA_STEPS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=DEVICE,
        )

        generate_submission(test_df["image_name"], predictions, SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
