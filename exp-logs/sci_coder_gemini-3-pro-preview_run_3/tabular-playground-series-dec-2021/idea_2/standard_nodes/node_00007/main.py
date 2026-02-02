import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Subset, DataLoader

# Import provided library functions
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import ResNetMLP, train_model, validate, predict_and_submit


def main():
    # 1. Configuration and Setup
    print("Initializing pipeline...")
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    BATCH_SIZE = 2048
    EPOCHS = 30
    LR = 1e-3
    PATIENCE = 7
    # Use full dataset for maximum performance
    TRAIN_SAMPLE_LIMIT = None

    # 2. Data Loading
    # We load full data first to ensure validation and test sets are complete
    # passing debug_limit=None to get_dataloaders ensures we get full val/test sets
    print("Loading datasets...")
    train_loader_full, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=4, load_cached_data=True, debug_limit=None
    )

    # Use full training set
    if TRAIN_SAMPLE_LIMIT is not None:
        print(f"Subsampling training data to {TRAIN_SAMPLE_LIMIT} samples...")
        train_dataset = train_loader_full.dataset
        if len(train_dataset) > TRAIN_SAMPLE_LIMIT:
            # Randomly select indices
            indices = torch.randperm(len(train_dataset))[:TRAIN_SAMPLE_LIMIT]
            train_subset = Subset(train_dataset, indices)
            train_loader = DataLoader(
                train_subset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
        else:
            train_loader = train_loader_full
    else:
        print("Using full training dataset...")
        train_loader = train_loader_full

    # 3. Model Initialization
    # Features: 13 numerical + 44 binary = 57
    # Classes: 6 (mapped classes)
    print("Initializing model...")
    # Cite solution_lesson_node_00006: Avoid arbitrarily scaling up model width and depth.
    # Reverting to compact model to prevent saturation and improve efficiency.
    model = ResNetMLP(
        input_dim=57,
        num_classes=6,
        num_blocks=3,
        hidden_dim=256,
        dropout_rate=0.2,
    )

    # 4. Training
    print("Starting training...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LR,
        patience=PATIENCE,
        device=device,
    )

    # 5. Validation Assessment
    print("Performing final validation...")
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis on Validation Set...")
    model.eval()
    all_inputs = []
    all_preds = []
    all_targets = []

    # Collect data (disable gradients for speed/memory)
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_inputs.append(inputs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    X_val = np.concatenate(all_inputs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Calculate Error (1 if wrong, 0 if correct)
    errors = (y_pred != y_true).astype(int)
    error_rate = errors.mean()
    print(f"Overall Error Rate: {error_rate:.4f}")

    # Feature Names Construction (matching logic in library/data_loader.py)
    num_cols = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Dist_Hydrology",
        "Vertical_Dist_Hydrology",
        "Horizontal_Dist_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Dist_Fire_Points",
        "Euclidean_Distance_To_Hydrology",
        "Elevation_Hydro",
        "Mean_Distance_Amenities",
    ]
    # 4 Wilderness Areas
    wild_cols = [f"Wilderness_Area{i+1}" for i in range(4)]
    # 40 Soil Types
    soil_cols = [f"Soil_Type{i+1}" for i in range(40)]

    feature_names = num_cols + wild_cols + soil_cols

    # Calculate Correlations
    print("Calculating feature-error correlations...")
    correlations = []
    for i in range(X_val.shape[1]):
        feat_col = X_val[:, i]
        # Handle constant features to avoid NaN
        if np.std(feat_col) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append(corr)

    # Display Top Correlations
    corr_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Correlation": correlations,
            "AbsCorrelation": np.abs(correlations),
        }
    )

    print("\nTop 10 Features associated with Error (by Absolute Correlation):")
    print(
        corr_df.sort_values("AbsCorrelation", ascending=False).head(10)[
            ["Feature", "Correlation"]
        ]
    )

    # 7. Submission
    if val_acc > 0.9620555555555556:
        print("\nGenerating submission for Test Set...")
        predict_and_submit(
            model, test_loader, output_path="./submission/submission.csv", device=device
        )
    else:
        print(
            f"\nValidation accuracy {val_acc} did not meet threshold 0.96205. Skipping submission."
        )
    print("Process complete.")


if __name__ == "__main__":
    main()
