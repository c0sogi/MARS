import os
import sys
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.data_loader import prepare_data, get_feature_cols, NFLDataset
from library.architecture import KinematicMLP
from library.engine import fit, evaluate, find_best_threshold, inference


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Using full training epochs
    print(f"Configuration: EPOCHS set to {Config.EPOCHS}.")

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Preparing Training Data...")
    train_dataset = prepare_data("train", load_cached_data=True)

    # Cite solution_lesson_node_00023: Prioritize data volume over architectural complexity.
    # Training on full dataset (3.4M samples) instead of subsampling.

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print("Preparing Validation Data...")
    val_dataset = prepare_data("validation", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = KinematicMLP().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cite solution_lesson_node_00008: Static Class Weighting vs. Focal Loss
    # Using Weighted BCE to handle extreme imbalance.
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Training
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # fit returns the model with best weights loaded and the best mcc score
    model, best_mcc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_path=save_path,
    )

    # 5. Validation & Failure Analysis
    print("\n=== Final Validation & Failure Analysis ===")

    # Evaluate on full validation set
    val_loss, val_y, val_probs = evaluate(model, val_loader, criterion, device)

    # Optimize threshold on the full validation set
    best_thresh, final_mcc = find_best_threshold(val_y, val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # Failure Analysis: Correlation between Error and Features
    print("Performing failure analysis...")
    errors = np.abs(val_y - val_probs)

    # Convert validation features to numpy for correlation calculation
    # val_dataset.X is a Tensor
    X_val_numpy = val_dataset.X.numpy()
    feature_names = get_feature_cols()

    # Create DataFrame for correlation
    # To save memory, we can compute correlation manually or use pandas on a subset if needed
    # Given 220GB RAM, pandas on full val set (860k rows) is trivial.
    df_analysis = pd.DataFrame(X_val_numpy, columns=feature_names)
    df_analysis["error_magnitude"] = errors

    # Compute correlation
    correlations = (
        df_analysis.corr()["error_magnitude"]
        .drop("error_magnitude")
        .abs()
        .sort_values(ascending=False)
    )

    print("\nTop 5 Features correlated with Prediction Error:")
    print(correlations.head(5))

    # 6. Submission
    TARGET_METRIC = 0.62458462731896

    if final_mcc > TARGET_METRIC:
        print(
            f"\nMetric ({final_mcc}) > Threshold ({TARGET_METRIC}). Generating submission..."
        )

        # Load Test Metadata (needed for contact_ids)
        test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
        df_test_meta = pd.read_csv(test_meta_path)

        # Load Test Data
        print("Preparing Test Data...")
        test_dataset = prepare_data("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Run Inference
        inference(model, test_loader, df_test_meta, device, best_thresh)

    else:
        print(
            f"\nMetric ({final_mcc}) did not beat threshold ({TARGET_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    main()
