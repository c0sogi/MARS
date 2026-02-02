import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.feature_engineering import FeatureEngineer
from library.dataset import ContactDataset
from library.model import EFWideResNet
from library.trainer import Trainer


def run_inference(model, loader, device):
    """
    Runs inference on a DataLoader.
    Returns probabilities and (optionally) labels.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            # Batch can be (x_cont, x_cat) or (x_cont, x_cat, labels)
            if len(batch) == 3:
                x_cont, x_cat, _ = batch
            else:
                x_cont, x_cat = batch

            x_cont = x_cont.to(device)
            x_cat = {k: v.to(device) for k, v in x_cat.items()}

            outputs = model(x_cont, x_cat)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())

    if all_probs:
        return np.concatenate(all_probs)
    return np.array([])


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Adjust Config for fast baseline execution
    Config.EPOCHS = 15
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Data Processing
    fe = FeatureEngineer(debug=Config.DEBUG)

    print("\n--- Processing Training Data ---")
    X_train, X_cat_train, y_train, _ = fe.process_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_TRACKING_PATH,
        is_train=True,
        load_cached_data=True,
    )

    print("\n--- Processing Validation Data ---")
    X_val, X_cat_val, y_val, _ = fe.process_dataset(
        Config.VAL_METADATA_PATH,
        Config.TRAIN_TRACKING_PATH,
        is_train=False,
        load_cached_data=True,
    )

    # Create Datasets and Loaders
    train_dataset = ContactDataset(X_train, X_cat_train, y_train)
    val_dataset = ContactDataset(X_val, X_cat_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    # Determine input dimension from features
    num_continuous = X_train.shape[1]
    print(f"\nModel Input Dimension: {num_continuous} continuous features + embeddings")

    model = EFWideResNet(num_continuous_features=num_continuous)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training
    trainer = Trainer(model, train_loader, val_loader, optimizer)
    best_threshold = trainer.fit()

    # 5. Final Validation Assessment
    print("\n--- Final Validation Assessment ---")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)

    # Run inference on validation set
    val_probs = run_inference(model, val_loader, Config.DEVICE)

    # Binarize and Compute MCC
    val_preds = (val_probs >= best_threshold).astype(int)
    final_mcc = compute_mcc(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_val - val_probs.flatten())

    # Compute correlation between error and continuous features
    # We use the scaled features X_val
    # Create a DataFrame for easy correlation computation
    feature_names = [f"feat_{i}" for i in range(num_continuous)]
    df_analysis = pd.DataFrame(X_val, columns=feature_names)
    df_analysis["error"] = errors

    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Error Magnitude:")
    print(correlations.head(5))
    print("\nBottom 5 Features correlated with Error Magnitude:")
    print(correlations.tail(5))

    # 7. Submission
    submission_threshold = 0.62458462731896

    if final_mcc > submission_threshold:
        print(
            f"\nValidation MCC ({final_mcc:.6f}) > Threshold ({submission_threshold}). Generating submission..."
        )

        # Process Test Data
        print("\n--- Processing Test Data ---")
        X_test, X_cat_test, _, test_ids = fe.process_dataset(
            Config.TEST_METADATA_PATH,
            Config.TEST_TRACKING_PATH,
            is_train=False,
            load_cached_data=True,
        )

        # Create Test Loader (y is None or dummy)
        # We pass dummy y to match Dataset signature, though it won't be used for metrics
        dummy_y = np.zeros(len(X_test))
        test_dataset = ContactDataset(X_test, X_cat_test, dummy_y)

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        test_probs = run_inference(model, test_loader, Config.DEVICE)

        # Apply Threshold
        test_preds = (test_probs >= best_threshold).astype(int)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"contact_id": test_ids, "contact": test_preds.flatten()}
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(
            f"Submission saved to {Config.SUBMISSION_PATH} with {len(submission_df)} rows."
        )

    else:
        print(
            f"\nValidation MCC ({final_mcc:.6f}) <= Threshold ({submission_threshold}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
