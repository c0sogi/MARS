import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import components from the provided libraries
from library.utils import set_seed, calculate_pos_weights, mixup_data, mixup_criterion
from library.model import BirdModel
from library.data_loader import BirdDataset


def train_fn(model, loader, criterion, optimizer, device, alpha):
    """
    Executes one epoch of training with Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply Mixup
        mixed_inputs, y_a, y_b, lam = mixup_data(inputs, targets, alpha, device)

        optimizer.zero_grad()
        outputs = model(mixed_inputs)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / dataset_size


def eval_fn(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Macro ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate Macro ROC AUC
    # Handle edge cases where a specific class might not be present in the validation fold
    auc_scores = []
    for i in range(all_targets.shape[1]):
        if len(np.unique(all_targets[:, i])) > 1:
            auc_scores.append(roc_auc_score(all_targets[:, i], all_preds[:, i]))

    if len(auc_scores) > 0:
        auc = np.mean(auc_scores)
    else:
        auc = 0.5

    return running_loss / dataset_size, auc


def inference_fn(model, loader, device):
    """
    Generates predictions for the test set.
    Returns raw probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())

    return np.vstack(all_preds)


def run_kfold_training(
    input_dir="./input",
    metadata_dir="./metadata",
    working_dir="./working/idea_5",
    submission_dir="./submission",
    num_folds=5,
    epochs=25,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-4,
    mixup_alpha=0.4,
    image_size=224,
    patience=5,
    seed=42,
):
    """
    Main execution function for Stratified K-Fold training and submission generation.
    """
    # Ensure output directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load and Prepare Data ---
    # Combine train and val splits from metadata to perform our own K-Fold split
    df_train_part = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    df_val_part = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    df_full_train = pd.concat([df_train_part, df_val_part], axis=0).reset_index(
        drop=True
    )

    df_test = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # Identify label columns
    label_cols = [c for c in df_full_train.columns if c.startswith("species_")]
    num_classes = len(label_cols)
    y_labels = df_full_train[label_cols].values

    # Create stratification targets based on label combinations (Power Set)
    # This ensures that unique combinations of species are distributed across folds
    y_str = ["".join(map(str, row.astype(int))) for row in y_labels]

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # Array to accumulate test predictions from each fold
    test_preds_sum = np.zeros((len(df_test), num_classes))

    # --- 2. K-Fold Loop ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_full_train, y_str)):
        print(f"\n=== Starting Fold {fold} ===")

        # Split DataFrames
        df_train_fold = df_full_train.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df_full_train.iloc[val_idx].reset_index(drop=True)

        # Initialize Datasets
        train_ds = BirdDataset(
            df_train_fold, input_dir, image_size=image_size, train=True
        )
        val_ds = BirdDataset(df_val_fold, input_dir, image_size=image_size, train=False)

        # Initialize Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        # Initialize Model
        model = MILResNet18(num_classes=num_classes, pretrained=True).to(device)

        # Calculate Class Weights for Loss
        y_train_tensor = torch.tensor(df_train_fold[label_cols].values)
        pos_weights = calculate_pos_weights(y_train_tensor, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training Variables
        best_auc = -1.0
        best_model_path = os.path.join(working_dir, f"model_fold_{fold}.pth")
        early_stop_counter = 0

        # --- 3. Training Epochs ---
        for epoch in range(epochs):
            train_loss = train_fn(
                model, train_loader, criterion, optimizer, device, mixup_alpha
            )
            val_loss, val_auc = eval_fn(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Fold {fold} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Checkpoint & Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Fold {fold} Best AUC: {best_auc}")

        # --- 4. Inference on Test Set ---
        # Load best model for this fold
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_ds = BirdDataset(df_test, input_dir, image_size=image_size, train=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        fold_preds = inference_fn(model, test_loader, device)
        test_preds_sum += fold_preds

    # --- 5. Generate Submission ---
    # Average predictions across folds
    avg_preds = test_preds_sum / num_folds

    submission_rows = []
    rec_ids = df_test["rec_id"].values

    for i, rec_id in enumerate(rec_ids):
        probs = avg_preds[i]
        for species_idx, prob in enumerate(probs):
            # ID format: rec_id * 100 + species_id
            row_id = int(rec_id * 100 + species_idx)
            submission_rows.append({"Id": row_id, "Probability": prob})

    sub_df = pd.DataFrame(submission_rows)
    sub_path = os.path.join(submission_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
