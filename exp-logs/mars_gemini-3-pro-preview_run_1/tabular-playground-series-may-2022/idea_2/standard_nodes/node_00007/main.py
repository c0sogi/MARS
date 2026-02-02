import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided libraries
from library.utils import seed_everything, get_device
from library.dataset import get_datasets
from library.model import ManufacturingNet
from library.trainer import Trainer


def get_feature_names():
    """
    Reconstructs feature names corresponding to X_num columns.
    Based on logic in library/preprocessing.py
    """
    train_path = "./metadata/train.csv"
    if os.path.exists(train_path):
        df = pd.read_csv(train_path, nrows=0)
        seq_col = "f_27"
        ignore_cols = ["id", "target", "source_path", seq_col]
        num_cols = [c for c in df.columns if c not in ignore_cols]
        # The preprocessing appends one feature: unique character count
        feature_names = num_cols + ["unique_char_count"]
        return feature_names
    return [f"feat_{i}" for i in range(100)]  # Fallback


def perform_failure_analysis(model, val_loader, device, feature_names):
    """
    Calculates correlation between model error and input features on validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_numericals = []

    print("\nPerforming Failure Analysis on Validation Set...")

    with torch.no_grad():
        for batch in val_loader:
            numerical = batch["numerical"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["label"].to(device)

            outputs = model(numerical, sequence)

            all_preds.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
            all_numericals.append(numerical.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_numericals = np.vstack(all_numericals)

    # Calculate Error
    errors = np.abs(all_targets - all_preds)

    # Create DataFrame for correlation calculation
    # Ensure feature names match dimensions
    if len(feature_names) != all_numericals.shape[1]:
        print(
            f"Warning: Feature name count ({len(feature_names)}) does not match data dimension ({all_numericals.shape[1]}). using indices."
        )
        feature_names = [f"feat_{i}" for i in range(all_numericals.shape[1])]

    df_features = pd.DataFrame(all_numericals, columns=feature_names)
    df_features["error_magnitude"] = errors

    # Calculate correlation with error
    correlations = df_features.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print("Top 10 Features correlated with Error Magnitude:")
    print(correlations.loc[correlations_abs.index[:10]])
    print("-" * 30)


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Configuration
    BATCH_SIZE = 4096  # Large batch size for A100
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    EMBEDDING_DIM = 48
    HIDDEN_UNITS = [512, 256, 128]
    DROPOUT_RATE = 0.2
    PATIENCE = 5
    THRESHOLD_AUC = 0.9914117319232936

    # 3. Load Data
    print("Loading datasets...")
    # Using full dataset to maximize score
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 4. Model Configuration
    num_numerical = train_ds.X_num.shape[1]
    seq_len = train_ds.X_seq.shape[1]

    max_idx_train = train_ds.X_seq.max()
    max_idx_val = val_ds.X_seq.max()
    max_idx_test = test_ds.X_seq.max()
    vocab_size = int(max(max_idx_train, max_idx_val, max_idx_test)) + 1

    print(
        f"Model Config: Num Features={num_numerical}, Seq Len={seq_len}, Vocab Size={vocab_size}"
    )

    # 5. Initialize Model
    model = ManufacturingNet(
        num_numerical_features=num_numerical,
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        seq_len=seq_len,
        hidden_units=HIDDEN_UNITS,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    # 6. Optimization
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 7. Training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        patience=PATIENCE,
    )

    trainer.fit(train_loader, val_loader, EPOCHS)

    # 8. Validation Metric
    final_metric = trainer.best_auc
    print(f"Final Validation Metric: {final_metric}")

    # 9. Failure Analysis
    # Load best model weights first (Trainer stores them but we need to ensure model has them)
    if trainer.best_model_state:
        model.load_state_dict(trainer.best_model_state)

    feature_names = get_feature_names()
    perform_failure_analysis(model, val_loader, device, feature_names)

    # 10. Submission
    if final_metric > THRESHOLD_AUC:
        print(
            f"Validation metric {final_metric} > threshold {THRESHOLD_AUC}. Generating submission..."
        )
        output_path = "./submission/submission.csv"
        trainer.predict(test_loader, output_path=output_path)
    else:
        print(
            f"Validation metric {final_metric} <= threshold {THRESHOLD_AUC}. Skipping submission."
        )


if __name__ == "__main__":
    main()
