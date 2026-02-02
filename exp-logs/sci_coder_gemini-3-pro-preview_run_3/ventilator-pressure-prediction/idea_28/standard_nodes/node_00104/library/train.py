import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, AverageMeter
from library.loss import MaskedL1Loss
from library.dataset import VentilatorDataset, add_features
from library.model import PCDRHNet


def load_data_and_create_loaders():
    """
    Loads data, applies feature engineering (with caching), reshapes for sequence modeling,
    scales features, and returns DataLoaders.
    """
    # Define cache paths
    cache_train_path = os.path.join(Config.WORKING_DIR, "train_eng.parquet")
    cache_val_path = os.path.join(Config.WORKING_DIR, "val_eng.parquet")
    cache_test_path = os.path.join(Config.WORKING_DIR, "test_eng.parquet")

    # 1. Load or Generate Features
    if (
        os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    ):
        print("Loading cached feature-engineered data...")
        train_df = pd.read_parquet(cache_train_path)
        val_df = pd.read_parquet(cache_val_path)
        test_df = pd.read_parquet(cache_test_path)
    else:
        print("Loading raw metadata and generating features...")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Apply provided feature engineering
        train_df = add_features(train_df)
        val_df = add_features(val_df)
        test_df = add_features(test_df)

        # Cache the processed dataframes
        print(f"Saving processed data to {Config.WORKING_DIR}...")
        train_df.to_parquet(cache_train_path, index=False)
        val_df.to_parquet(cache_val_path, index=False)
        test_df.to_parquet(cache_test_path, index=False)

    # 2. Scaling Stream A Features
    print("Scaling Stream A features...")
    scaler = RobustScaler()
    # Fit only on training data
    train_A_flat = train_df[Config.STREAM_A_FEATURES].values
    scaler.fit(train_A_flat)

    # 3. Reshaping and Dataset Creation Helper
    def process_and_reshape(df, is_test=False):
        # Transform Stream A
        data_A = scaler.transform(df[Config.STREAM_A_FEATURES].values)
        # Extract Stream B
        data_B = df[Config.STREAM_B_FEATURES].values

        # Reshape to (Num_Breaths, 80, Features)
        # We assume the data is sorted by breath_id and time_step (guaranteed by metadata generation)
        # and that each breath has exactly 80 time steps.
        if len(df) % 80 != 0:
            raise ValueError(f"Data length {len(df)} is not divisible by 80.")

        data_A = data_A.reshape(-1, 80, data_A.shape[1])
        data_B = data_B.reshape(-1, 80, data_B.shape[1])

        target = None
        if not is_test and Config.TARGET_COL in df.columns:
            target = df[Config.TARGET_COL].values
            target = target.reshape(-1, 80)

        return data_A, data_B, target

    print("Reshaping data for sequence model...")
    X_train_a, X_train_b, y_train = process_and_reshape(train_df)
    X_val_a, X_val_b, y_val = process_and_reshape(val_df)
    X_test_a, X_test_b, _ = process_and_reshape(test_df, is_test=True)

    # 4. Create Datasets
    train_dataset = VentilatorDataset(X_train_a, X_train_b, y_train)
    val_dataset = VentilatorDataset(X_val_a, X_val_b, y_val)
    test_dataset = VentilatorDataset(X_test_a, X_test_b, None)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,  # Shuffle breaths
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_df


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_meter = AverageMeter()

    for x_a, x_b, y in loader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(x_a)

        # Compute loss
        # x_b is (Batch, 80, 1), squeeze to (Batch, 80) for broadcasting with pred/y
        loss = criterion(pred, y, x_b.squeeze(-1))

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        optimizer.step()

        loss_meter.update(loss.item(), x_a.size(0))

    return loss_meter.avg


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for x_a, x_b, y in loader:
            x_a = x_a.to(device)
            x_b = x_b.to(device)
            y = y.to(device)

            pred = model(x_a)

            # Compute metric (masked MAE)
            loss = criterion(pred, y, x_b.squeeze(-1))

            loss_meter.update(loss.item(), x_a.size(0))

    return loss_meter.avg


def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for x_a, x_b, _ in loader:
            x_a = x_a.to(device)
            pred = model(x_a)
            preds.append(pred.cpu().numpy())

    # Concatenate all batches: (N_breaths, 80)
    return np.concatenate(preds, axis=0)


def run_training():
    # Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, test_df = load_data_and_create_loaders()

    # Model
    input_dim = len(Config.STREAM_A_FEATURES)
    print(f"Initializing PCDRHNet with input_dim={input_dim}")
    model = PCDRHNet(input_dim).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
        verbose=True,
    )

    criterion = MaskedL1Loss()

    # Training Loop
    best_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model. Val Loss: {val_loss:.6f}")

    # Inference on Test Set
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating predictions...")
    predictions = predict(model, test_loader, device)

    # Flatten predictions to match submission format (N_breaths * 80)
    predictions_flat = predictions.flatten()

    # Ensure submission matches test_df order
    # test_df is the dataframe used to create the test dataset, so order is preserved
    submission = pd.DataFrame(
        {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: predictions_flat}
    )

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
