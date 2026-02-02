import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import soundfile as sf
from torch.utils.data import DataLoader

# Import library modules
from library.config import CFG
from library.utils import set_seed
from library.dataset import AudioDataset, collate_fn
from library.model import AudioModel
from library.engine import train_fn, valid_fn, inference_fn, save_submission


def main():
    # 1. Setup
    set_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override CFG for fast baseline execution as per requirements
    # Reducing epochs to 15 (half of original) to ensure completion within 2 hours
    # while maintaining enough capacity to reach the target score.
    CFG.epochs = 15

    # Ensure working directory exists
    os.makedirs(CFG.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    val_df = pd.read_csv(CFG.VAL_CSV)
    test_df = pd.read_csv(CFG.TEST_CSV)

    # Datasets
    train_dataset = AudioDataset(train_df, mode="train")
    val_dataset = AudioDataset(val_df, mode="val")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.inference_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # 3. Model Initialization
    model = AudioModel(pretrained=True)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
    )

    # 5. Training Loop
    best_score = 0.0
    best_model_path = os.path.join(CFG.WORKING_DIR, "best_model.pth")

    for epoch in range(CFG.epochs):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_score = valid_fn(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Validation Metric
    print(f"Final Validation Metric: {best_score}")

    # 7. Failure Analysis
    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Generate predictions on validation set for analysis
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            val_preds_list.append(probs.cpu().numpy())
            val_targets_list.append(targets.numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    # Calculate Error Magnitude (Mean Absolute Error per sample)
    # Shape: (N_samples,)
    sample_errors = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Extract Metadata Features
    # Duration
    durations = []
    for idx, row in val_df.iterrows():
        fpath = os.path.join(CFG.INPUT_ROOT, row["filepath"])
        try:
            # Fast header read
            info = sf.info(fpath)
            durations.append(info.duration)
        except:
            durations.append(0.0)
    durations = np.array(durations)

    # Label Count
    label_counts = val_df[CFG.target_columns].sum(axis=1).values

    # Calculate Correlations
    corr_duration = np.corrcoef(sample_errors, durations)[0, 1]
    corr_labels = np.corrcoef(sample_errors, label_counts)[0, 1]

    print(f"Correlation between Error and Duration: {corr_duration}")
    print(f"Correlation between Error and Label Count: {corr_labels}")

    # 8. Submission
    TARGET_THRESHOLD = 0.855624439013878

    if best_score > TARGET_THRESHOLD:
        # Load Test Dataset
        test_dataset = AudioDataset(test_df, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.inference_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Inference
        predictions = inference_fn(model, test_loader, device)

        # Save Submission
        save_submission(predictions, test_df, "./submission/submission.csv")


if __name__ == "__main__":
    main()
