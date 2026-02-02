import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import MGMTNet
from library.engine import train_one_epoch, evaluate, predict_test_set


def get_validation_predictions(model, dataloader, device):
    """
    Runs inference on the validation set to get raw predictions and targets
    for failure analysis.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, targets, subject_ids in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Input shape: (B, C, H, W)
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Flatten predictions
            batch_probs = probs.view(-1)

            all_preds.extend(batch_probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_ids.extend(subject_ids.numpy())

    return np.array(all_ids), np.array(all_targets), np.array(all_preds)


def perform_failure_analysis(val_ids, val_targets, val_preds):
    """
    Analyzes the correlation between prediction error and input data characteristics.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Error
    errors = np.abs(val_targets - val_preds)

    # Create Analysis DataFrame
    df_analysis = pd.DataFrame(
        {
            "BraTS21ID": val_ids,
            "target": val_targets,
            "prediction": val_preds,
            "error": errors,
        }
    )

    # Extract metadata features for these subjects
    # We'll count files in the directories to see if depth affects performance
    metadata_features = []

    # Load validation metadata to get paths
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    # Map ID to paths
    path_map = df_val_meta.set_index("BraTS21ID").to_dict("index")

    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    for sid in val_ids:
        feats = {"BraTS21ID": sid}
        if sid in path_map:
            row = path_map[sid]
            for mod in modalities:
                # Construct full path
                rel_path = row.get(f"{mod.lower()}_path", "")
                full_path = os.path.join(Config.INPUT_DIR, rel_path)

                # Count files
                count = 0
                if os.path.exists(full_path):
                    count = len(
                        [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                    )

                feats[f"{mod}_count"] = count
        metadata_features.append(feats)

    df_features = pd.DataFrame(metadata_features)

    # Merge
    df_analysis = pd.merge(df_analysis, df_features, on="BraTS21ID", how="left")

    # Compute Correlations
    feature_cols = [c for c in df_analysis.columns if c.endswith("_count")]
    if feature_cols:
        correlations = (
            df_analysis[feature_cols + ["error"]].corr()["error"].drop("error")
        )
        print("Correlation between Error Magnitude and Modality Slice Counts:")
        print(correlations.sort_values(ascending=False))
    else:
        print("Could not extract features for correlation analysis.")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print(f"Initializing {Config.MODEL_NAME}...")
    model = MGMTNet(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        in_chans=Config.NUM_CHANNELS,
        drop_rate=Config.DROPOUT_RATE,
    )
    model = model.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Validation & Metric
    print("\nLoading best model for final evaluation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Re-evaluate to confirm metric
    _, final_auc = evaluate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    val_ids, val_targets, val_preds = get_validation_predictions(
        model, val_loader, device
    )
    perform_failure_analysis(val_ids, val_targets, val_preds)

    # 8. Submission
    THRESHOLD = 0.6705454545454544

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        df_submission = predict_test_set(model, test_loader, device)

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_auc}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
