import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats

# Import library functions
from library.utils import CompetitionMetric, haversine_distance
from library.model import DSTResNet
from library.data_loader import get_dataloaders
from library.train import train_model
from library.inference import generate_submission

# Configuration
WINDOW_SIZE = 11
BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 1e-3
PATIENCE = 3
HIDDEN_DIM = 128
DYNAMIC_FEATURES = 6
STATIC_FEATURES = 2
CHECKPOINT_DIR = "./working/idea_3"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
VAL_META_CACHE = os.path.join(CHECKPOINT_DIR, "val_meta.parquet")
VAL_X_CACHE = os.path.join(CHECKPOINT_DIR, "val_X.npy")
GT_PATH = "./metadata/validation_metadata.csv"


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_validation_and_analysis():
    print("\nStarting Validation and Failure Analysis...")

    # 1. Load Validation Data and Metadata
    if not os.path.exists(VAL_META_CACHE) or not os.path.exists(VAL_X_CACHE):
        print("Validation cache not found. Ensure training ran successfully.")
        return float("inf")

    df_val_meta = pd.read_parquet(VAL_META_CACHE)
    X_val = np.load(VAL_X_CACHE)

    # 2. Load Ground Truth
    if not os.path.exists(GT_PATH):
        print(f"Ground truth file not found at {GT_PATH}")
        return float("inf")

    df_gt = pd.read_csv(GT_PATH)

    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSTResNet(
        dynamic_features=DYNAMIC_FEATURES,
        static_features=STATIC_FEATURES,
        window_size=WINDOW_SIZE,
        hidden_dim=HIDDEN_DIM,
    )

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Model checkpoint not found at {BEST_MODEL_PATH}")
        return float("inf")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 4. Inference
    # Create DataLoader for validation inference
    from library.model import GNSSWindowDataset
    from torch.utils.data import DataLoader

    val_dataset = GNSSWindowDataset(X_val)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    preds_list = []
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_list.append(outputs.cpu().numpy())

    if not preds_list:
        print("No predictions generated.")
        return float("inf")

    preds_enu = np.concatenate(preds_list, axis=0)

    # 5. Reconstruct Coordinates
    # Extract baseline WLS coordinates from metadata
    wls_lat = df_val_meta["WlsLat"].values
    wls_lon = df_val_meta["WlsLon"].values

    d_east = preds_enu[:, 0]
    d_north = preds_enu[:, 1]

    # Local linear approximation
    r_earth = 6378137.0
    d_lat_rad = d_north / r_earth
    d_lon_rad = d_east / (r_earth * np.cos(np.radians(wls_lat)))

    pred_lat = wls_lat + np.degrees(d_lat_rad)
    pred_lon = wls_lon + np.degrees(d_lon_rad)

    # Add predictions to metadata dataframe
    df_val_meta["LatitudeDegrees_pred"] = pred_lat
    df_val_meta["LongitudeDegrees_pred"] = pred_lon

    # 6. Merge with Ground Truth
    # Ensure we match on tripId and UnixTimeMillis
    # Rename GT columns for merging if necessary, but CompetitionMetric expects specific names
    # CompetitionMetric expects df_pred and df_gt.
    # df_pred needs: tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees
    # df_gt needs: tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees

    df_pred_formatted = df_val_meta[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees_pred", "LongitudeDegrees_pred"]
    ].copy()
    df_pred_formatted.rename(
        columns={
            "LatitudeDegrees_pred": "LatitudeDegrees",
            "LongitudeDegrees_pred": "LongitudeDegrees",
        },
        inplace=True,
    )

    # 7. Calculate Metric
    score = CompetitionMetric(df_pred_formatted, df_gt)
    print(f"Final Validation Metric: {score}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Merge predictions with GT to calculate errors per row
    df_analysis = pd.merge(
        df_pred_formatted,
        df_gt[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_pred", "_gt"),
    )

    # Calculate Haversine distance error
    df_analysis["error_meters"] = haversine_distance(
        df_analysis["LatitudeDegrees_pred"],
        df_analysis["LongitudeDegrees_pred"],
        df_analysis["LatitudeDegrees_gt"],
        df_analysis["LongitudeDegrees_gt"],
    )

    # Correlate error with input features
    # Features in X_val: [dLat, dLon, dAlt, MeanCn0, MeanUnc, SatCount, WlsLat, WlsLon]
    # Indices: 0, 1, 2, 3, 4, 5, 6, 7
    # We take the center of the window for analysis
    center_idx = WINDOW_SIZE // 2

    # Extract features (N, Features)
    features_center = X_val[:, center_idx, :]

    feature_names = [
        "dLat",
        "dLon",
        "dAlt",
        "MeanCn0",
        "MeanUnc",
        "SatCount",
        "WlsLat",
        "WlsLon",
    ]

    print("Spearman Correlation between Error Magnitude and Input Features:")
    for i, name in enumerate(feature_names):
        feat_values = features_center[:, i]
        # Ensure alignment (X_val corresponds to df_val_meta which corresponds to df_analysis after merge?
        # Yes, because val_loader is sequential and df_val_meta is what we predicted on.
        # However, merge with GT might drop rows if GT is missing some timestamps.
        # But process_data ensures we only keep rows present in metadata.
        # Let's align by index just to be safe, assuming df_analysis preserves order or we re-index.
        # Actually, df_analysis is a merge result, order might change.
        # Better strategy: Add features to df_val_meta before merge.
        pass

    # Add features to df_val_meta for robust alignment
    for i, name in enumerate(feature_names):
        df_val_meta[name] = features_center[:, i]

    # Re-merge for analysis with features
    df_analysis_feat = pd.merge(
        df_val_meta,
        df_gt[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
        on=["tripId", "UnixTimeMillis"],
        suffixes=("", "_gt"),
    )

    # Recalculate error just to be sure
    df_analysis_feat["error_meters"] = haversine_distance(
        df_analysis_feat["LatitudeDegrees_pred"],
        df_analysis_feat["LongitudeDegrees_pred"],
        df_analysis_feat[
            "LatitudeDegrees"
        ],  # GT from merge usually overwrites or suffixes?
        # Wait, df_val_meta doesn't have GT Lat/Lon.
        # The merge brings in LatitudeDegrees from GT (if suffix applied correctly).
        # Let's check suffixes.
        df_analysis_feat["LongitudeDegrees"],  # This is from GT
    )

    # Calculate correlations
    correlations = {}
    for name in feature_names:
        corr, _ = scipy.stats.spearmanr(
            df_analysis_feat["error_meters"], df_analysis_feat[name]
        )
        correlations[name] = corr

    # Sort and print
    for name, corr in sorted(
        correlations.items(), key=lambda item: abs(item[1]), reverse=True
    ):
        print(f"  {name}: {corr:.4f}")

    return score


def main():
    set_seed(42)

    print("==================================================")
    print("DST-ResNet Orchestration Script")
    print("==================================================")

    # 1. Train Model
    print("\n[Step 1] Training Model...")
    train_model(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        window_size=WINDOW_SIZE,
        load_cached_data=True,
    )

    # 2. Validate and Analyze
    print("\n[Step 2] Validating and Analyzing...")
    metric = run_validation_and_analysis()

    # 3. Generate Submission
    threshold = 4.256982128481356
    if metric < threshold:
        print(
            f"\n[Step 3] Metric {metric:.6f} is better than threshold {threshold:.6f}. Generating submission..."
        )
        generate_submission(
            batch_size=BATCH_SIZE,
            window_size=WINDOW_SIZE,
            checkpoint_path=BEST_MODEL_PATH,
            output_path="./submission/submission.csv",
            load_cached_data=True,
            hidden_dim=HIDDEN_DIM,
        )
    else:
        print(
            f"\n[Step 3] Metric {metric:.6f} did not meet threshold {threshold:.6f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
