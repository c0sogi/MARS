import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import soundfile as sf

# Import from provided library files
from library.config import TrainConfig
from library.preprocess import cache_data
from library.dataset import (
    get_balanced_dataloader,
    get_test_dataloader,
    IDX2LABEL,
    LABEL2IDX,
)
from library.model import SKResNetConformer
from library.engine import train_one_epoch, validate, set_seed


def main():
    # --- 1. Configuration & Setup ---
    # Override config for fast baseline execution
    TrainConfig.epochs = 10  # Sufficient for convergence with pretrained backbone
    TrainConfig.batch_size = 64  # Utilize A100 memory
    TrainConfig.debug = False  # Use full dataset to meet accuracy threshold

    # Ensure directories exist and set seeds
    TrainConfig.setup_directories()
    set_seed(TrainConfig.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Preprocessing & Caching ---
    # Generate cached features if not present
    cache_data(load_cached_data=True)

    # --- 3. Data Loading ---
    train_loader = get_balanced_dataloader(
        TrainConfig.train_metadata_path, TrainConfig.batch_size, is_training=True
    )
    val_loader = get_balanced_dataloader(
        TrainConfig.val_metadata_path, TrainConfig.batch_size, is_training=False
    )

    # --- 4. Model Initialization ---
    model = SKResNetConformer().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=TrainConfig.lr, weight_decay=TrainConfig.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TrainConfig.epochs, eta_min=TrainConfig.min_lr
    )

    # --- 5. Training Loop ---
    best_acc = 0.0

    for epoch in range(TrainConfig.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), TrainConfig.model_save_path)

    # --- 6. Final Evaluation ---
    # Load best model for inference
    if os.path.exists(TrainConfig.model_save_path):
        model.load_state_dict(
            torch.load(TrainConfig.model_save_path, map_location=device)
        )
    model.eval()

    # Collect all validation predictions for analysis
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    final_acc = np.mean(all_preds == all_targets)
    print(f"Final Validation Metric: {final_acc}")

    # --- 7. Failure Analysis ---
    val_df = pd.read_csv(TrainConfig.val_metadata_path)

    # Ensure dataframe matches predictions (val_loader is not shuffled)
    if len(val_df) != len(all_preds):
        val_df = val_df.iloc[: len(all_preds)]

    val_df["pred"] = all_preds
    val_df["target"] = all_targets
    val_df["error"] = (val_df["pred"] != val_df["target"]).astype(int)

    # Extract durations for correlation analysis
    durations = []
    for filepath in val_df["filepath"]:
        full_path = os.path.join(TrainConfig.input_dir, filepath)
        try:
            info = sf.info(full_path)
            durations.append(info.duration)
        except:
            durations.append(1.0)
    val_df["duration"] = durations

    # Map labels to indices
    val_df["label_idx"] = val_df["label"].map(
        lambda x: LABEL2IDX.get(x, LABEL2IDX["unknown"])
    )

    # Compute correlations
    if val_df["duration"].std() > 1e-6:
        corr_dur = np.corrcoef(val_df["error"], val_df["duration"])[0, 1]
    else:
        corr_dur = 0.0

    if val_df["label_idx"].std() > 1e-6:
        corr_lbl = np.corrcoef(val_df["error"], val_df["label_idx"])[0, 1]
    else:
        corr_lbl = 0.0

    print(f"Correlation (Error vs Duration): {corr_dur}")
    print(f"Correlation (Error vs LabelIdx): {corr_lbl}")

    # --- 8. Submission ---
    threshold = 0.9832324978392394
    if final_acc > threshold:
        test_loader = get_test_dataloader(
            TrainConfig.test_metadata_path, TrainConfig.batch_size
        )
        predictions = []

        with torch.no_grad():
            for inputs, filepaths in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)

                preds_cpu = preds.cpu().numpy()
                for p, fp in zip(preds_cpu, filepaths):
                    fname = os.path.basename(fp)
                    label = IDX2LABEL[p]
                    predictions.append({"fname": fname, "label": label})

        sub_df = pd.DataFrame(predictions)
        os.makedirs(os.path.dirname(TrainConfig.submission_path), exist_ok=True)
        sub_df.to_csv(TrainConfig.submission_path, index=False)


if __name__ == "__main__":
    main()
