import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef

import library.config as config
import library.data_utils as data_utils
import library.dataset as dataset
import library.model as model
import library.train_utils as train_utils


def main():
    # 1. Setup
    print("Setting up...")
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # Load dataframes (cached if available)
    df_train = data_utils.load_and_process_data("train")
    df_val = data_utils.load_and_process_data("val")
    df_test = data_utils.load_and_process_data("test")

    # 3. Subsampling removed to use full dataset (Cite solution_lesson_node_00023)

    # 4. Scaling
    print("Scaling features...")
    # Fit scaler on (subsampled) train, transform all
    X_train, X_val, X_test, scaler = data_utils.scale_data(df_train, df_val, df_test)

    # 5. Dataset & DataLoader Creation
    print("Creating Datasets and Loaders...")

    # Prepare targets and ground indicators
    y_train = df_train["contact"].values
    g_train = df_train["is_ground"].values

    y_val = df_val["contact"].values
    g_val = df_val["is_ground"].values

    # Test has no targets usually, but the file has a placeholder 'contact' column
    g_test = df_test["is_ground"].values

    train_dataset = dataset.ContactDataset(X_train, y_train, g_train)
    val_dataset = dataset.ContactDataset(X_val, y_val, g_val)
    test_dataset = dataset.ContactDataset(X_test, None, g_test)

    # Loaders
    # Train: Shuffle=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val/Test: Shuffle=False (Crucial for alignment in failure analysis/submission)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 6. Model Initialization
    print("Initializing Model...")
    net = model.KinematicMLP(input_dim=config.INPUT_DIM)
    net.to(device)

    # 7. Training
    print("Starting Training...")
    # Using fewer epochs for fast baseline execution
    FAST_EPOCHS = 5
    trained_model = train_utils.train_model(
        net,
        train_loader,
        val_loader,
        epochs=FAST_EPOCHS,
        lr=config.LEARNING_RATE,
        patience=config.EARLY_STOPPING_PATIENCE,
        pos_weight=config.POS_WEIGHT,
        save_path=config.MODEL_SAVE_PATH,
    )

    # 8. Threshold Optimization
    print("Optimizing Threshold...")
    best_threshold = train_utils.optimize_threshold(trained_model, val_loader)

    # 9. Final Validation Metric
    print("Computing Final Validation Metrics...")
    # Re-run evaluation with the best threshold
    criterion = torch.nn.BCEWithLogitsLoss()  # Placeholder for eval
    val_loss, _, val_probs, val_targets = train_utils.evaluate(
        trained_model, val_loader, criterion, device
    )

    val_preds = (val_probs > best_threshold).astype(int)
    final_mcc = matthews_corrcoef(val_targets, val_preds)

    print(f"Final Validation Metric: {final_mcc}")

    # 10. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_probs)

    # Ensure alignment
    if len(errors) != len(df_val):
        print("Warning: Length mismatch in failure analysis. Skipping correlation.")
    else:
        # Add error to dataframe temporarily
        analysis_df = df_val.copy()
        analysis_df["error"] = errors

        # Select features to correlate (using center frame features from lags)
        features_to_check = [
            "distance_lag_0",
            "speed_1_lag_0",
            "speed_2_lag_0",
            "acceleration_1_lag_0",
            "acceleration_2_lag_0",
            "is_ground",
        ]

        # Check if these columns exist
        available_feats = [f for f in features_to_check if f in analysis_df.columns]

        print("Correlation between Error and Features:")
        if available_feats:
            corrs = (
                analysis_df[available_feats + ["error"]].corr()["error"].drop("error")
            )
            print(corrs)
        else:
            print("Could not find suitable features for correlation analysis.")

    # 11. Submission
    TARGET_METRIC = 0.62458462731896
    if final_mcc > TARGET_METRIC:
        print(
            f"Validation MCC ({final_mcc}) > Target ({TARGET_METRIC}). Generating submission..."
        )
        train_utils.generate_submission(
            trained_model,
            test_loader,
            threshold=best_threshold,
            output_path=config.SUBMISSION_PATH,
        )
    else:
        print(
            f"Validation MCC ({final_mcc}) <= Target ({TARGET_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
