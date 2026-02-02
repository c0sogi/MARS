import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import log_loss

# Import provided library modules
import library.config as config
import library.utils as utils
import library.model as model_lib
import library.data_loader as data_loader_lib
import library.trainer as trainer_lib

# ==================================================================================
# 1. CONFIGURATION & SETUP
# ==================================================================================
# Adjust hyperparameters for speed and stability within the time limit
config.MAX_EPOCHS = 50
config.PATIENCE = 10

# Set seeds for reproducibility
utils.set_seed(config.SEED)

# ==================================================================================
# 2. DATA SPLITTING & MONKEY PATCHING
# ==================================================================================
# We need to ensure the training loop ONLY sees the data in metadata/train.csv
# and leaves metadata/val.csv as a strict hold-out set.
# Since the library functions are pre-written to load the full file, we monkey-patch
# the load_data function.

print("Loading metadata for split enforcement...")
train_meta_df = pd.read_csv(config.TRAIN_META_PATH)
val_meta_df = pd.read_csv(config.VAL_META_PATH)

train_ids = set(train_meta_df["id"].values)
val_ids = set(val_meta_df["id"].values)

# Store original function to call internally
original_load_data = model_lib.load_data


def masked_load_data(debug=False):
    """
    Wrapper around the original load_data that filters the training data
    to include ONLY samples present in metadata/train.csv.
    Test data and stats are returned as is.
    """
    train_raw, test_raw, stats, inc_mean = original_load_data(debug)

    # Filter training data
    filtered_train = [item for item in train_raw if item["id"] in train_ids]

    return filtered_train, test_raw, stats, inc_mean


# Apply the patch to both modules where load_data is used/defined
model_lib.load_data = masked_load_data
data_loader_lib.load_data = masked_load_data

print(f"Monkey patch applied. Training will use {len(train_ids)} samples.")
print(f"Validation will use {len(val_ids)} hold-out samples.")

# ==================================================================================
# 3. TRAINING
# ==================================================================================
print("\nStarting Training Pipeline...")
# This will run 5-fold CV on the 'filtered_train' data
trainer_lib.run_training(debug=False)

# ==================================================================================
# 4. HOLD-OUT VALIDATION
# ==================================================================================
print("\nStarting Hold-out Validation...")

# Retrieve the raw data again (using original loader to get everything)
full_train_raw, _, stats, inc_mean = original_load_data(debug=False)

# Extract the hold-out validation set
val_data_raw = [item for item in full_train_raw if item["id"] in val_ids]

# Create Dataset and DataLoader for Hold-out set
val_ds = model_lib.IcebergDataset(
    val_data_raw, stats, augment=False, inc_angle_mean=inc_mean
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
)

# Load all 5 trained models
models = []
for fold in range(config.NUM_FOLDS):
    path = os.path.join(config.MODEL_DIR, f"model_fold_{fold}.pth")
    if os.path.exists(path):
        m = model_lib.RDP_WBN().to(config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=config.DEVICE))
        m.eval()
        models.append(m)
    else:
        print(f"Warning: Model for fold {fold} missing.")

if not models:
    print("No models trained. Aborting.")
    sys.exit(1)

# Inference Loop
all_preds = []
all_targets = []
all_ids = []
all_inc_angles = []

with torch.no_grad():
    for imgs, metas, targets, ids in val_loader:
        imgs = imgs.to(config.DEVICE)
        metas_dev = metas.to(config.DEVICE)

        # Ensemble Prediction
        batch_preds = []
        for m in models:
            p = m(imgs, metas_dev)
            batch_preds.append(p.cpu().numpy())

        # Average probabilities
        avg_p = np.mean(batch_preds, axis=0)

        all_preds.extend(avg_p.flatten())
        all_targets.extend(targets.numpy().flatten())
        all_ids.extend(ids)
        all_inc_angles.extend(metas.numpy().flatten())

# Calculate Metric
y_true = np.array(all_targets)
y_pred = np.array(all_preds)

# Numerical stability for log loss
y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
final_metric = log_loss(y_true, y_pred_clipped)

print(f"Final Validation Metric: {final_metric}")

# ==================================================================================
# 5. FAILURE ANALYSIS
# ==================================================================================
print("\nPerforming Failure Analysis...")
errors = np.abs(y_true - y_pred)
df_analysis = pd.DataFrame(
    {
        "id": all_ids,
        "error": errors,
        "inc_angle": all_inc_angles,
        "target": y_true,
        "prediction": y_pred,
    }
)

# Correlation with Incidence Angle
# Note: inc_angle is already imputed in the dataset, so no NaNs here
corr_inc = df_analysis["error"].corr(df_analysis["inc_angle"])
print(f"Correlation between Error Magnitude and Incidence Angle: {corr_inc}")

# Identify worst failures
print("Top 5 Worst Predictions:")
print(
    df_analysis.sort_values("error", ascending=False).head(5)[
        ["id", "target", "prediction", "error", "inc_angle"]
    ]
)

# ==================================================================================
# 6. SUBMISSION
# ==================================================================================
THRESHOLD = 0.14772333549413377

if final_metric < THRESHOLD:
    print(
        f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
    )
    # predict_and_submit uses load_data to get test set.
    # Our patched load_data returns the full test set, so this is safe.
    model_lib.predict_and_submit()
else:
    print(
        f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
    )
