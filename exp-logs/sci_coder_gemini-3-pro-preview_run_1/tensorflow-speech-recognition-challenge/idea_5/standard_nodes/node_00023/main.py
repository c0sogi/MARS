import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import get_dataset
from library.model import get_model
from library.train import train_one_epoch, validate
from library.utils import set_seed


def main():
    # 1. Configuration and Setup
    config = Config()
    # Override for fast baseline execution as requested
    config.EPOCHS = 45
    config.BATCH_SIZE = 64  # Increase batch size for A100 efficiency

    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    # 2. Data Loading
    # Load cached balanced training data if available
    train_dataset = get_dataset("train", config=config, load_cached_data=True)
    val_dataset = get_dataset("val", config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = get_model(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        val_metrics = validate(model, val_loader, criterion, device, config)

        scheduler.step()

        # Checkpoint
        val_acc = val_metrics.metrics["Accuracy"]["avg"]
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)

    # 5. Final Evaluation & Failure Analysis
    # Load best model
    if os.path.exists(config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Run inference on validation set to get element-wise results
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            _, preds = torch.max(output, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate Final Metric
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    final_acc = np.mean(all_preds == all_targets)

    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis: Correlation between Error and Duration
    val_df = val_dataset.df.copy()
    val_df["pred"] = all_preds
    val_df["target"] = all_targets
    val_df["error"] = (val_df["pred"] != val_df["target"]).astype(int)

    # Calculate durations
    durations = []
    for idx, row in val_df.iterrows():
        path = os.path.join(config.INPUT_ROOT, row["filepath"])
        try:
            # Use torchaudio.info for fast metadata reading
            info = torchaudio.info(path)
            duration = info.num_frames / info.sample_rate
            durations.append(duration)
        except:
            durations.append(0.0)

    val_df["duration"] = durations

    # Calculate Correlation
    if val_df["duration"].std() > 0:
        corr = val_df["error"].corr(val_df["duration"])
        print(f"Correlation between error and duration: {corr}")
    else:
        print("Correlation between error and duration: Undefined (constant duration)")

    # 6. Submission
    THRESHOLD = 0.9754180602006689

    if final_acc > THRESHOLD:
        test_dataset = get_dataset("test", config=config)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)
                _, preds = torch.max(output, 1)
                test_preds.extend(preds.cpu().numpy())

        # Map IDs to Labels
        pred_labels = [config.ID2LABEL[p] for p in test_preds]
        fnames = test_dataset.df["filepath"].apply(os.path.basename).tolist()

        # Save
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        df_sub = pd.DataFrame({"fname": fnames, "label": pred_labels})
        df_sub.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
