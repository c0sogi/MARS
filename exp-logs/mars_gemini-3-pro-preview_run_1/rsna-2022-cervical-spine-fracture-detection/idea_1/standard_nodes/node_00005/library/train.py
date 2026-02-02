import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import FractureSliceDataset, get_transforms, prepare_training_data
from library.model import FractureClassifier
from library.utils import weighted_log_loss, sort_filenames_numerically


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_validation_slice_df(load_cached_data=True):
    """
    Generates a dataframe for validation containing slices for all validation studies
    using the defined stride to mimic inference.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "val_slice_df.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    df_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    samples = []
    stride = Config.INFERENCE_STRIDE

    for idx, row in df_meta.iterrows():
        uid = row["StudyInstanceUID"]
        path = os.path.join(Config.TRAIN_IMAGES_DIR, uid)

        if not os.path.exists(path):
            continue

        try:
            files = [f for f in os.listdir(path) if f.endswith(".dcm")]
        except OSError:
            continue

        files = sort_filenames_numerically(files)

        # Apply Stride
        for i in range(0, len(files), stride):
            fname = files[i]
            slice_num = int(os.path.splitext(fname)[0])
            samples.append({"StudyInstanceUID": uid, "slice_number": slice_num})

    df = pd.DataFrame(samples)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_DATASET_SIZE)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Define weights matching the competition metric
    # C1-C7: 1.0, patient_overall: 7.0
    # Cite {solution_lesson_node_00002}: Maintaining 512x512 resolution, but improving loss focus.
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0], device=device)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # Calculate unreduced loss
        loss_unreduced = criterion(outputs, labels)

        # Apply weights and mean
        loss = (loss_unreduced * weights).mean()

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size if dataset_size > 0 else 0.0


def validate(model, val_loader, val_metadata_df, device):
    model.eval()
    results = []

    # Inference on validation slices
    with torch.no_grad():
        for images, uids, slice_nums in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.cpu().numpy()

            # Collect results
            # uids is a tuple of strings
            for i in range(len(uids)):
                row = {"StudyInstanceUID": uids[i]}
                for idx, col in enumerate(Config.TARGET_COLS):
                    row[col] = preds[i][idx]
                results.append(row)

    if not results:
        return float("inf")

    # Aggregate predictions: Max pooling per study
    df_pred = pd.DataFrame(results)
    df_pred_agg = (
        df_pred.groupby("StudyInstanceUID")[Config.TARGET_COLS].max().reset_index()
    )

    # Merge with Ground Truth
    # val_metadata_df contains the true labels
    df_merge = pd.merge(
        val_metadata_df, df_pred_agg, on="StudyInstanceUID", suffixes=("_true", "_pred")
    )

    if df_merge.empty:
        return float("inf")

    # Prepare arrays for metric calculation
    # Ensure columns are in the correct order as expected by weighted_log_loss
    y_true = df_merge[[f"{c}_true" for c in Config.TARGET_COLS]].values
    y_pred = df_merge[[f"{c}_pred" for c in Config.TARGET_COLS]].values

    score = weighted_log_loss(y_true, y_pred)
    return score


def run_training():
    seed_everything(Config.SEED)

    # --- Data Loading ---
    train_df = prepare_training_data(load_cached_data=True)
    val_slice_df = prepare_validation_slice_df(load_cached_data=True)
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    # --- Datasets & Loaders ---
    train_dataset = FractureSliceDataset(
        train_df,
        Config.TRAIN_IMAGES_DIR,
        transform=get_transforms("train"),
        is_test=False,
    )

    # For validation, we use is_test=True to retrieve StudyInstanceUIDs for aggregation
    val_dataset = FractureSliceDataset(
        val_slice_df,
        Config.TRAIN_IMAGES_DIR,
        transform=get_transforms("val"),
        is_test=True,
    )

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

    # --- Model Setup ---
    device = torch.device(Config.DEVICE)
    model = FractureClassifier(pretrained=Config.PRETRAINED).to(device)

    # Loss: Binary Cross Entropy for training (slice level)
    # Use reduction='none' to apply custom class weights in train_one_epoch
    criterion = nn.BCELoss(reduction="none")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # --- Training Loop ---
    best_loss = float("inf")
    patience = 3
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, val_metadata, device)

        scheduler.step()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss}")

        # Checkpoint and Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Validation Loss: {best_loss}")
