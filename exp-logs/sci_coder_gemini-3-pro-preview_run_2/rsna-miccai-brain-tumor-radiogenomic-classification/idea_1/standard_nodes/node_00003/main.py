import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train
import library.inference as inference


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Reproducibility
    # -------------------------------------------------------------------------
    utils.seed_everything(config.SEED)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    # Running with config epochs
    train.run_training(epochs=config.EPOCHS)

    # -------------------------------------------------------------------------
    # 3. Validation Assessment
    # -------------------------------------------------------------------------
    # Load validation metadata
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    # Initialize validation dataset and loader
    val_dataset = data.BraTSDataset(
        metadata=df_val,
        base_dir=config.INPUT_DIR,
        transform=data.get_transforms("val"),
        is_test=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the best model checkpoint
    device = config.DEVICE
    net = model.MGMTNet(pretrained=False)
    net = net.to(device)

    if os.path.exists(config.CHECKPOINT_PATH):
        state_dict = torch.load(config.CHECKPOINT_PATH, map_location=device)
        net.load_state_dict(state_dict)
    else:
        # Fallback if training failed to produce a checkpoint (unlikely)
        pass

    net.eval()

    all_targets = []
    all_probs = []

    # Run inference on validation set
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            logits = net(inputs)
            probs = torch.sigmoid(logits)

            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy().flatten())

    # Calculate and print the required metric
    val_auc = roc_auc_score(all_targets, all_probs)
    print(f"Final Validation Metric: {val_auc}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    # Calculate error magnitude
    targets_np = np.array(all_targets)
    probs_np = np.array(all_probs)
    errors = np.abs(targets_np - probs_np)

    # Extract simple meta-features (slice counts) to check for correlation with error
    # This helps identify if 'less data' (fewer slices) leads to higher error.
    meta_features = []
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    for idx, row in df_val.iterrows():
        feat = {"error": errors[idx]}
        for mod in modalities:
            # Construct path
            dir_path = os.path.join(config.INPUT_DIR, row[f"path_{mod}"])
            # Count files (robustly)
            try:
                if os.path.exists(dir_path):
                    # Fast count
                    count = len([name for name in os.listdir(dir_path)])
                else:
                    count = 0
            except Exception:
                count = 0
            feat[f"{mod}_slices"] = count
        meta_features.append(feat)

    df_analysis = pd.DataFrame(meta_features)

    # Compute correlations
    if not df_analysis.empty:
        correlations = df_analysis.corr()["error"].drop("error")
        print(
            "Failure Analysis - Correlation between Error Magnitude and Input Features:"
        )
        print(correlations)

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    # Only generate submission if validation metric exceeds the threshold
    if val_auc > 0.5696363636363637:
        inference.generate_submission()
    else:
        print(f"Validation AUC {val_auc} did not exceed baseline. Skipping submission.")


if __name__ == "__main__":
    main()
