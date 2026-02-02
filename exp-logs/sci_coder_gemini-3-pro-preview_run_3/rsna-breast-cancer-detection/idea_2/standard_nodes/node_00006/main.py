import sys
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import EfficientNetV2Classifier
from library.engine import (
    train_representation_epoch,
    train_calibration_epoch,
    evaluate,
    generate_submission,
    EarlyStopping,
)


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Override Config for Fast Baseline execution (as per requirements)
    # Reducing epochs to ensure completion within strict time limits while
    # leveraging the speed of the A100 GPU.
    Config.STAGE1_EPOCHS = 3
    Config.STAGE2_EPOCHS = 3

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------------------
    model = EfficientNetV2Classifier(
        backbone_name=Config.BACKBONE_NAME, pretrained=True
    )
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    # -------------------------------------------------------------------------
    # 3. Stage 1: Representation Learning (Balanced)
    # -------------------------------------------------------------------------
    # Load data with caching enabled
    loaders_s1 = get_dataloaders(stage=1, debug=Config.DEBUG, load_cached_data=True)

    optimizer_s1 = torch.optim.AdamW(
        model.parameters(), lr=Config.STAGE1_LR, weight_decay=Config.STAGE1_WEIGHT_DECAY
    )
    early_stopper_s1 = EarlyStopping(patience=2, path=Config.STAGE1_CHECKPOINT)

    for epoch in range(Config.STAGE1_EPOCHS):
        train_loss = train_representation_epoch(
            model, loaders_s1["train"], optimizer_s1, device, criterion
        )
        val_loss, val_pf1 = evaluate(model, loaders_s1["val"], device, criterion)

        early_stopper_s1(val_loss, model)
        if early_stopper_s1.early_stop:
            break

    # Load best model from Stage 1
    if os.path.exists(Config.STAGE1_CHECKPOINT):
        model.load_state_dict(torch.load(Config.STAGE1_CHECKPOINT, map_location=device))

    # -------------------------------------------------------------------------
    # 4. Stage 2: Probability Calibration (Natural)
    # -------------------------------------------------------------------------
    loaders_s2 = get_dataloaders(stage=2, debug=Config.DEBUG, load_cached_data=True)

    # Freeze backbone and reset head for calibration
    model.freeze_backbone()
    model.reset_classifier()
    model.to(device)

    # Optimize only the trainable parameters (head)
    optimizer_s2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=Config.STAGE2_LR
    )
    early_stopper_s2 = EarlyStopping(patience=2, path=Config.STAGE2_CHECKPOINT)

    for epoch in range(Config.STAGE2_EPOCHS):
        train_loss = train_calibration_epoch(
            model, loaders_s2["train"], optimizer_s2, device, criterion
        )
        val_loss, val_pf1 = evaluate(model, loaders_s2["val"], device, criterion)

        early_stopper_s2(val_loss, model)
        if early_stopper_s2.early_stop:
            break

    # Load best model from Stage 2
    if os.path.exists(Config.STAGE2_CHECKPOINT):
        model.load_state_dict(torch.load(Config.STAGE2_CHECKPOINT, map_location=device))

    # -------------------------------------------------------------------------
    # 5. Final Validation
    # -------------------------------------------------------------------------
    # Evaluate on the full validation set
    val_loss, final_pf1 = evaluate(model, loaders_s2["val"], device, criterion)

    # REQUIRED PRINT: Final Validation Metric
    print(f"Final Validation Metric: {final_pf1}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    # Extract predictions and targets for correlation analysis
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loaders_s2["val"]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate absolute error
    errors = np.abs(all_targets - all_preds)

    # Load validation metadata to correlate errors with features
    # Using the parquet cache if available to match data loader logic, else CSV
    cache_path_val = os.path.join(Config.WORKING_DIR, "val_meta.parquet")
    if os.path.exists(cache_path_val):
        df_val = pd.read_parquet(cache_path_val)
    else:
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    if Config.DEBUG:
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)

    # Safety check for length alignment
    min_len = min(len(df_val), len(errors))
    df_val = df_val.iloc[:min_len]
    errors = errors[:min_len]

    df_val["error"] = errors

    # Features to analyze
    features_to_analyze = ["age", "density", "view", "laterality", "machine_id"]
    correlations = {}

    for feat in features_to_analyze:
        if feat in df_val.columns:
            # Handle categorical features
            if df_val[feat].dtype == "object":
                le = LabelEncoder()
                series = df_val[feat].fillna("Unknown").astype(str)
                encoded = le.fit_transform(series)
                corr = np.corrcoef(encoded, df_val["error"])[0, 1]
            else:
                # Handle numerical features
                series = df_val[feat].fillna(df_val[feat].median())
                corr = np.corrcoef(series, df_val["error"])[0, 1]
            correlations[feat] = corr

    print("Error Correlations with Features:")
    for k, v in correlations.items():
        print(f"{k}: {v}")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.06310755014419556

    if final_pf1 > THRESHOLD:
        generate_submission(model, loaders_s2["test"], device)


if __name__ == "__main__":
    main()
