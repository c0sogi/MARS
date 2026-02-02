import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

# Import from provided library files
from library.utils import set_seed, get_device
from library.data_loader import get_dataloaders
from library.model import SiameseNetwork
from library.train import train_one_epoch, validate
from library.predict import generate_submission

# ==========================================
# Configuration
# ==========================================
CACHE_DIR = "./working/idea_40/"
TRAIN_META = "./metadata/train.parquet"
VAL_META = "./metadata/val.parquet"
TEST_META = "./metadata/test.parquet"
SUBMISSION_PATH = "./submission/submission.csv"
BEST_MODEL_PATH = os.path.join(CACHE_DIR, "best_model.pth")

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
SEED = 42
SUBMISSION_THRESHOLD = 0.6978181818181817


def main():
    # 1. Setup
    set_seed(SEED)
    device = get_device()
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 2. Data Loading
    # We load train and val loaders. Test loader is handled in generate_submission if needed.
    train_loader, val_loader, _ = get_dataloaders(
        train_meta_path=TRAIN_META,
        val_meta_path=VAL_META,
        test_meta_path=TEST_META,
        batch_size=BATCH_SIZE,
        num_workers=4,
        load_cached_data=True,
        cache_dir=CACHE_DIR,
    )

    # 3. Model Initialization
    model = SiameseNetwork(
        model_name="efficientnet_b0", pretrained=True, drop_path_rate=0.2
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training Loop
    best_auc = 0.0

    for epoch in range(EPOCHS):
        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        # We silence per-epoch logs to keep output clean as per instructions,
        # or print minimal info if needed. The instructions say "Only print the required information".
        # However, monitoring progress is usually acceptable. We will keep it minimal.
        # print(f"Epoch {epoch+1}: Train AUC {train_auc:.4f}, Val AUC {val_auc:.4f}")

    # 5. Final Evaluation & Metric
    # Load best model
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    model.eval()

    # We need detailed predictions for failure analysis, so we run a manual inference loop on val
    val_preds = []
    val_targets = []
    val_ids = []  # We need to map back to metadata

    # To map IDs, we need to load them. The loader doesn't return IDs by default.
    # We will assume the loader order matches the metadata order if shuffle=False (which it is for val).
    # We'll load metadata dataframe to get features and IDs.
    df_val = pd.read_parquet(VAL_META)

    with torch.no_grad():
        for batch in val_loader:
            xe, xo, labels = batch
            xe, xo = xe.to(device), xo.to(device)
            logits = model(xe, xo).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.extend(probs)
            val_targets.extend(labels.numpy())

    final_auc = roc_auc_score(val_targets, val_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\nFailure Analysis:")

    # Calculate errors
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    errors = np.abs(val_preds - val_targets)

    # Extract meta-features from dataframe
    # We calculate slice counts for each modality
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    meta_features = {}

    for mod in modalities:
        col = f"{mod}_paths"
        # Handle potential None or empty lists
        counts = df_val[col].apply(lambda x: len(x) if x is not None else 0)
        meta_features[f"{mod}_count"] = counts.values

    # Ensure lengths match (loader might drop last batch if drop_last=True, but default is False)
    # The provided loader code doesn't set drop_last=True, so lengths should match.
    if len(errors) != len(df_val):
        # Fallback: truncate to minimum length if mismatch occurs (unlikely)
        min_len = min(len(errors), len(df_val))
        errors = errors[:min_len]
        for k in meta_features:
            meta_features[k] = meta_features[k][:min_len]

    # Calculate Correlations
    print("Correlation between Error Magnitude and Input Features:")
    for name, feature_values in meta_features.items():
        # Point-Biserial isn't appropriate here (continuous vs continuous), using Pearson
        corr, _ = stats.pearsonr(errors, feature_values)
        print(f"{name}: {corr:.4f}")

    # 7. Conditional Submission
    if final_auc > SUBMISSION_THRESHOLD:
        generate_submission(
            test_meta_path=TEST_META,
            model_path=BEST_MODEL_PATH,
            submission_path=SUBMISSION_PATH,
            cache_dir=CACHE_DIR,
            batch_size=BATCH_SIZE,
            num_workers=4,
            load_cached_data=True,
        )


if __name__ == "__main__":
    main()
