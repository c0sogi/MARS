import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Import from provided library files
from library.config import Config, ECGRN, FocalLoss, ContactDataset, get_data, set_seed
from library.utils import compute_mcc


def run():
    # 1. Configuration Setup
    # Adjust config for fast baseline execution
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 4096
    Config.DEBUG = False  # Ensure we load the full dataset structure initially

    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # get_data loads train and validation sets combined, distinguished by 'is_val' column
    df_all = get_data(mode="train", load_cached_data=True)

    # Define feature columns
    exclude_cols = ["contact_id", "game_play", "step", "contact", "is_val"]
    feature_cols = [c for c in df_all.columns if c not in exclude_cols]
    Config.INPUT_DIM = len(feature_cols)

    # Split into Train and Validation
    # Validation set must be the full hold-out set from metadata/validation.csv
    train_df = df_all[df_all["is_val"] == 0]
    val_df = df_all[df_all["is_val"] == 1]

    # Limit training samples for fast baseline
    # We use 1,000,000 samples which is sufficient for convergence but faster than full 3.4M
    if len(train_df) > 1000000:
        train_df = train_df.sample(n=1000000, random_state=Config.SEED)

    # 3. Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    y_train = train_df["contact"].values.astype(float)

    # Transform validation set using the same scaler
    X_val = scaler.transform(val_df[feature_cols].values)
    y_val = val_df["contact"].values.astype(float)

    # Create DataLoaders
    train_loader = DataLoader(
        ContactDataset(X_train, y_train),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ContactDataset(X_val, y_val),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Model Initialization
    model = ECGRN(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        dropout=Config.DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # 5. Training Loop
    for epoch in range(Config.EPOCHS):
        model.train()
        for X_b, y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

    # 6. Validation and Threshold Optimization
    model.eval()
    val_probs_list = []

    with torch.no_grad():
        for X_b, _ in val_loader:
            X_b = X_b.to(device, non_blocking=True)
            probs = model(X_b)
            val_probs_list.append(probs.cpu().numpy())

    val_probs = np.concatenate(val_probs_list).flatten()

    # Optimize Threshold
    best_threshold = 0.5
    best_mcc = -1.0

    # Check thresholds from 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, 81)
    for t in thresholds:
        mcc = compute_mcc(y_val, val_probs, threshold=t)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = t

    print(f"Final Validation Metric: {best_mcc:.16f}")

    # 7. Failure Analysis
    # Calculate correlation between error magnitude and input features
    errors = np.abs(y_val - val_probs)

    # We use X_val (scaled numpy array) for correlation
    correlations = []

    for i, feature_name in enumerate(feature_cols):
        # Pearson correlation
        if np.std(X_val[:, i]) > 1e-9:  # Avoid division by zero for constant features
            corr = np.corrcoef(X_val[:, i], errors)[0, 1]
            correlations.append((feature_name, corr))
        else:
            correlations.append((feature_name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nFailure Analysis - Top 5 Features Correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")

    # 8. Submission Generation
    target_metric = 0.62458462731896

    if best_mcc > target_metric:
        df_test = get_data(mode="test", load_cached_data=True)

        # Align features
        for c in feature_cols:
            if c not in df_test.columns:
                df_test[c] = 0

        X_test = scaler.transform(df_test[feature_cols].values)

        test_loader = DataLoader(
            ContactDataset(X_test),
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        test_probs_list = []
        with torch.no_grad():
            for X_b in test_loader:
                X_b = X_b.to(device, non_blocking=True)
                probs = model(X_b)
                test_probs_list.append(probs.cpu().numpy())

        test_probs = np.concatenate(test_probs_list).flatten()

        submission_df = pd.DataFrame(
            {
                "contact_id": df_test["contact_id"],
                "contact": (test_probs > best_threshold).astype(int),
            }
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)


if __name__ == "__main__":
    run()
