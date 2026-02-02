import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import BraTSDataset
from library.network import SHDNet
from library.engine import train_model, validate, predict


def main():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Data Preparation
    # The Dataset class handles caching automatically in Config.CACHE_DIR
    # Update cache names to force regeneration with new NUM_SLICES and IMG_SIZE
    train_dataset = BraTSDataset(
        metadata_path=Config.TRAIN_META_PATH,
        cache_name="train_v2",
        load_cached_data=True,
        is_train=True,
    )

    val_dataset = BraTSDataset(
        metadata_path=Config.VAL_META_PATH,
        cache_name="val_v2",
        load_cached_data=True,
        is_train=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Model Initialization
    model = SHDNet(drop_path_rate=Config.DROP_PATH_RATE)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=Config.EPOCHS,
        save_path=Config.MODEL_PATH,
    )

    # 5. Final Validation
    # Load the best saved model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    model.eval()

    # Collect predictions and targets for analysis
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Targets from loader are tensors
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metric
    final_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    errors = np.abs(all_targets - all_preds)

    # Load metadata to correlate errors with input features (e.g., slice counts)
    val_df = pd.read_parquet(Config.VAL_META_PATH)

    # Extract meta-features
    meta_features = {}
    for mod in ["flair", "t1w", "t1wce", "t2w"]:
        col_name = f"{mod}_paths"
        # Count number of slices/files available for each modality
        counts = val_df[col_name].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
        )
        meta_features[f"{mod}_count"] = counts.values

    print("Correlation between Error Magnitude and Input Features:")
    for name, values in meta_features.items():
        if len(values) == len(errors):
            corr = np.corrcoef(values, errors)[0, 1]
            print(f"{name}: {corr}")
        else:
            print(f"{name}: Length mismatch, skipping.")

    # 7. Submission
    SUBMISSION_THRESHOLD = 0.6978181818181817

    if final_auc > SUBMISSION_THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")

        test_dataset = BraTSDataset(
            metadata_path=Config.TEST_META_PATH,
            cache_name="test_v2",
            load_cached_data=True,
            is_train=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # predict returns (ids, probabilities) for test set
        test_ids, test_preds = predict(model, test_loader, device)

        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_preds})

        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"Validation AUC ({final_auc}) <= Threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
