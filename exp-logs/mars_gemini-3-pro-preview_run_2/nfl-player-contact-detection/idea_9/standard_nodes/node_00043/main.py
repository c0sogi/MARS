import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.data_processing import DataProcessor
from library.dataset import NFLDataset
from library.model import WIRKNet
from library.training import FocalLoss, train_epoch, evaluate, optimize_threshold


def main():
    # 1. Configuration Overrides for Fast Baseline
    # We reduce epochs to ensure runtime constraints are met
    Config.MAX_EPOCHS = 3

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    processor = DataProcessor()
    # Load full data (cached if available)
    X_train, y_train, X_val, y_val = processor.get_train_val_data(load_cached_data=True)

    # 3. Subsample Training Data for Speed
    # We strictly limit training samples for the fast baseline requirement
    # but keep the full validation set for the required metric calculation.
    TRAIN_SUBSET_SIZE = 100000
    if len(X_train) > TRAIN_SUBSET_SIZE:
        print(f"Subsampling training data to {TRAIN_SUBSET_SIZE} samples...")
        indices = np.random.choice(len(X_train), TRAIN_SUBSET_SIZE, replace=False)
        X_train_sub = X_train.iloc[indices].reset_index(drop=True)
        y_train_sub = y_train.iloc[indices].reset_index(drop=True)
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    print(f"Training samples: {len(X_train_sub)}")
    print(f"Validation samples: {len(X_val)}")

    # 4. Dataset & DataLoader
    train_dataset = NFLDataset(X_train_sub, y_train_sub)
    val_dataset = NFLDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Double batch size for inference
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 5. Model Initialization
    # Determine input dimensions
    num_cont = train_dataset.X_cont.shape[1]
    num_cat = train_dataset.X_cat.shape[1]

    # Calculate vocab sizes dynamically
    cat_vocab_sizes = []
    if num_cat > 0:
        for i in range(num_cat):
            # Calculate max index across both train and val to ensure embedding coverage
            max_train = train_dataset.X_cat[:, i].max().item()
            max_val = val_dataset.X_cat[:, i].max().item()
            cat_vocab_sizes.append(max(max_train, max_val) + 1)

    print(
        f"Initializing WIRK-Net with {num_cont} continuous features and vocab sizes {cat_vocab_sizes}"
    )

    model = WIRKNet(
        num_cont_features=num_cont,
        cat_vocab_sizes=cat_vocab_sizes,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_RESIDUAL_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    criterion = FocalLoss(alpha=Config.FOCAL_LOSS_ALPHA, gamma=Config.FOCAL_LOSS_GAMMA)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 6. Training Loop
    best_mcc = -1.0
    best_threshold = 0.5

    print("Starting training...")
    for epoch in range(Config.MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_targets = evaluate(
            model, val_loader, criterion, device
        )

        # Flatten arrays for metric calculation
        val_targets_flat = val_targets.flatten()
        val_probs_flat = val_probs.flatten()

        curr_thresh, curr_mcc = optimize_threshold(val_targets_flat, val_probs_flat)

        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | Train Loss: {train_loss:.4f} | Val MCC: {curr_mcc:.4f}"
        )

        if curr_mcc > best_mcc:
            best_mcc = curr_mcc
            best_threshold = curr_thresh
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 7. Final Evaluation Output
    # Required format: Final Validation Metric: <value>
    print(f"Final Validation Metric: {best_mcc}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load best model for analysis
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()

    # Get predictions on validation set
    _, val_probs, val_targets = evaluate(model, val_loader, criterion, device)
    val_probs_flat = val_probs.flatten()
    val_targets_flat = val_targets.flatten()

    # Calculate absolute errors
    errors = np.abs(val_targets_flat - val_probs_flat)

    # Create analysis dataframe
    # We use X_val which is already a DataFrame
    analysis_df = X_val.copy()
    analysis_df["error"] = errors

    # Compute correlations between features and error
    # Select only numeric columns for correlation
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    correlations = (
        analysis_df[numeric_cols]
        .corrwith(analysis_df["error"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Prediction Error:")
    print(correlations.head(5))

    # 9. Submission
    TARGET_METRIC = 0.62458462731896
    if best_mcc > TARGET_METRIC:
        print(
            f"\nValidation MCC ({best_mcc}) > Target ({TARGET_METRIC}). Generating submission..."
        )

        # Load test data
        X_test, ids_test = processor.get_test_data(load_cached_data=True)

        test_dataset = NFLDataset(X_test, None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Predict
        all_probs = []
        model.eval()
        with torch.no_grad():
            for x_cat, x_cont in test_loader:
                x_cat = x_cat.to(device)
                x_cont = x_cont.to(device)
                logits = model(x_cat, x_cont)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        all_probs = np.concatenate(all_probs).flatten()

        # Apply optimized threshold
        predictions = (all_probs >= best_threshold).astype(int)

        submission = pd.DataFrame({"contact_id": ids_test, "contact": predictions})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation MCC ({best_mcc}) did not meet target ({TARGET_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
