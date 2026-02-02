import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time

# Import provided library functions
from library.utils import seed_everything, get_device, save_submission
from library.data_processing import load_and_preprocess_data
from library.model import ParallelDCNResNeXt
from library.train import train_one_epoch, validate, predict


def main():
    # 1. Setup & Configuration
    seed_everything(42)
    device = get_device()
    BASE_DIR = "./working/idea_11"

    # 2. Data Loading
    # We use the full dataset to ensure we can hit the high metric threshold.
    # The A100 GPU is sufficient to process this efficiently.
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = load_and_preprocess_data(
        batch_size=4096, load_cached_data=True, base_dir=BASE_DIR, sample_size=None
    )

    # Determine input dimensions
    sample_input, _ = next(iter(train_loader))
    input_dim = sample_input.shape[1]
    num_classes = 7  # Classes 1-7 mapped to 0-6 internally

    # 3. Model Initialization
    print("Initializing model...")
    model = ParallelDCNResNeXt(
        input_dim=input_dim,
        num_classes=num_classes,
        dcn_layers=3,
        resnext_layers=3,
        d_model=1024,
        cardinality=32,
    ).to(device)

    # 4. Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )

    # 5. Training Loop
    epochs = 20  # fast baseline limit
    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    epochs_no_improve = 0

    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training finished in {time.time() - start_time:.2f}s")

    # 6. Final Validation & Metric
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_val_loss, final_val_acc = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_acc}")

    # 7. Failure Analysis
    print("Performing failure analysis on validation set...")
    model.eval()
    all_preds = []
    all_targets = []
    all_inputs = []

    # Collect validation predictions and inputs
    # We move inputs to CPU to save GPU memory and for numpy correlation
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs_dev = inputs.to(device)
            outputs = model(inputs_dev)
            _, preds = outputs.max(1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            all_inputs.append(inputs.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_inputs = np.concatenate(all_inputs)

    # Calculate Error Vector (1 if wrong, 0 if correct)
    errors = (all_preds != all_targets).astype(int)

    # Reconstruct feature names for meaningful output
    # Logic mirrors library.utils._feature_engineering and get_data
    try:
        meta_df = pd.read_parquet("./metadata/train.parquet")
        raw_cols = [c for c in meta_df.columns if c not in ["Id", "Cover_Type"]]
        # Add engineered features
        eng_cols = [
            "Aspect_Sin",
            "Aspect_Cos",
            "Hydrology_Distance_Euclidean",
            "Hydrology_Elevation_Abs",
            "Mean_Distance_Amenities",
        ]
        full_cols = raw_cols + eng_cols

        # Sort into Continuous and Binary (as done in utils.py)
        bin_names = [
            c
            for c in full_cols
            if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
        ]
        cont_names = [c for c in full_cols if c not in bin_names]
        feature_names = cont_names + bin_names
    except Exception as e:
        print(f"Warning: Could not reconstruct feature names ({e}). Using indices.")
        feature_names = [f"Feature_{i}" for i in range(input_dim)]

    # Calculate Correlations
    print("Calculating feature correlations with error magnitude...")
    correlations = []
    for i in range(input_dim):
        feat_col = all_inputs[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feat_col) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append(
            (feature_names[i] if i < len(feature_names) else f"Feat_{i}", corr)
        )

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.6f}")

    # 8. Conditional Submission
    THRESHOLD = 0.9625041666666667
    if final_val_acc > THRESHOLD:
        print(
            f"Metric ({final_val_acc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        raw_preds = predict(model, test_loader, device)

        # Map 0-6 back to 1-7
        final_preds = raw_preds + 1

        save_submission(
            test_ids, final_preds, output_path="./submission/submission.csv"
        )
    else:
        print(
            f"Metric ({final_val_acc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
