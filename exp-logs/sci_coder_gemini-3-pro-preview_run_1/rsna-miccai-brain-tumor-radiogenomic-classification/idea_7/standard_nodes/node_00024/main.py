import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import model
from library import train


def perform_failure_analysis(net, val_loader, device):
    """
    Analyzes model errors on the validation set by correlating error magnitude
    with target class and input file counts.
    """
    print("\n=== Failure Analysis ===")

    net.eval()
    all_probs = []
    all_targets = []
    all_ids = []

    # Run inference on validation set
    with torch.no_grad():
        for images, targets, ids in val_loader:
            images = images.to(device)
            outputs = net(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_ids.append(ids.numpy())

    all_probs = np.concatenate(all_probs).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_ids = np.concatenate(all_ids).flatten()

    # Calculate absolute error
    errors = np.abs(all_targets - all_probs)

    # Create analysis DataFrame
    df_analysis = pd.DataFrame(
        {
            "BraTS21ID": all_ids,
            "target": all_targets,
            "prob": all_probs,
            "error": errors,
        }
    )

    # Load metadata to retrieve file paths for feature extraction
    df_val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    df_analysis = pd.merge(df_analysis, df_val_meta, on="BraTS21ID", how="left")

    # Extract 'flair_count' as a metadata feature (proxy for scan resolution/volume)
    flair_counts = []
    for _, row in df_analysis.iterrows():
        try:
            # Metadata paths are relative to input dir
            p = os.path.join(config.INPUT_DIR, row["flair_path"])
            if os.path.exists(p):
                flair_counts.append(len(os.listdir(p)))
            else:
                flair_counts.append(0)
        except Exception:
            flair_counts.append(0)

    df_analysis["flair_count"] = flair_counts

    # Calculate correlations
    corr_target = df_analysis["error"].corr(df_analysis["target"])
    corr_flair = df_analysis["error"].corr(df_analysis["flair_count"])

    print(f"Correlation (Error vs Target Class): {corr_target:.15f}")
    print(f"Correlation (Error vs FLAIR Slice Count): {corr_flair:.15f}")


def generate_submission_file(net, test_loader, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("\n=== Generating Submission ===")

    net.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = net(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.append(probs)
            all_ids.append(ids.numpy())

    all_probs = np.concatenate(all_probs).flatten()
    all_ids = np.concatenate(all_ids).flatten()

    # Create submission DataFrame
    df_sub = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_probs})

    # Ensure output directory exists
    os.makedirs("submission", exist_ok=True)
    submission_path = "submission/submission.csv"

    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Running on device: {device}")

    # 2. Train Model
    # library.train.run_training handles data loading, training loop, and saving best_model.pth
    # It returns the best validation AUC achieved.
    best_auc = train.run_training(load_cached=True)

    # 3. Report Final Metric
    print(f"Final Validation Metric: {best_auc}")

    # 4. Load Best Model for Inference
    # We need to reload the model state that achieved the best AUC
    net = model.MontageEfficientNet(
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        num_classes=config.NUM_CLASSES,
        drop_rate=config.DROPOUT_RATE,
    )
    checkpoint_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    utils.load_checkpoint(checkpoint_path, net, device=device)
    net.to(device)

    # Retrieve DataLoaders (cached) for validation analysis and testing
    _, val_loader, test_loader = data_loader.get_dataloaders(load_cached=True)

    # 5. Failure Analysis
    perform_failure_analysis(net, val_loader, device)

    # 6. Submission Logic
    # Threshold defined in the task description
    THRESHOLD = 0.6705454545454544

    if best_auc > THRESHOLD:
        generate_submission_file(net, test_loader, device)
    else:
        print(f"Metric {best_auc} <= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
