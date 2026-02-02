import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig
import gc

# Ensure library modules are accessible
sys.path.append(os.getcwd())

from library.config import Config
from library import pipeline, utils, data, model_head

# ==============================================================================
# 1. Configuration & Setup
# ==============================================================================
# Optimize for speed and memory within the 2-hour limit
# Training DeBERTa-Large for 1 epoch on 12k samples takes ~15 mins on A100.
# 5 folds * 15 mins = 75 mins, which fits comfortably within 2 hours.
Config.epochs = 1
Config.n_folds = 5
Config.batch_size = 8  # Increased for A100 (40GB)
Config.accum_iter = 2  # Adjusted to keep effective batch size reasonable
Config.lgbm_params["n_estimators"] = 1000  # Reduce from 5000 for faster stacking
Config.lgbm_params["early_stopping_rounds"] = 50

# Ensure reproducibility
utils.seed_everything(Config.seed)

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================
print("Starting Cross-Validation Pipeline...")
pipeline.run_cv()

# ==============================================================================
# 3. Validation Assessment
# ==============================================================================
print("\nPerforming Validation Assessment...")

# Load the validation metadata to identify hold-out samples
val_meta_df = pd.read_csv(Config.val_metadata_path)
val_ids = set(val_meta_df["essay_id"].values)

# Load the full processed training data (which includes validation data)
# We need this to get the features and true scores aligned with OOF predictions
tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
full_dataset = data.load_processed_data(tokenizer, mode="train", load_cached_data=True)

# Reconstruct OOF Embeddings from Cache
# run_cv saves these to cache but doesn't return them directly.
cache_manager = utils.CacheManager(Config.cache_dir)
backbone_config = AutoConfig.from_pretrained(Config.model_name)
backbone_hidden_size = backbone_config.hidden_size
oof_embeddings = np.zeros((len(full_dataset), backbone_hidden_size), dtype=np.float32)

folds = data.get_folds(full_dataset, n_folds=Config.n_folds, seed=Config.seed)
for fold, (train_idx, val_idx) in enumerate(folds):
    fold_config = {"fold": fold, "model": Config.model_name, "debug": Config.debug}
    cached_oof = cache_manager.load(
        f"oof_embeddings_fold_{fold}", config_dict=fold_config
    )

    if cached_oof is not None:
        oof_embeddings[val_idx] = cached_oof
    else:
        print(
            f"Warning: OOF embeddings for fold {fold} not found in cache. Validation metric may be inaccurate."
        )

# Load Trained Stacking Model
stacker = model_head.StackingTrainer()
stacker_path = os.path.join(Config.output_dir, "lgbm_stacking.txt")
stacker.load(stacker_path)

# Generate Predictions on the full dataset (using OOF embeddings)
# The stacker uses both embeddings and meta-features
all_preds = stacker.predict(oof_embeddings, full_dataset.meta_features)

# Filter predictions for the hold-out validation set
val_preds = []
val_trues = []
val_indices = []

# Create mappings for efficient lookup
id_to_pred = {eid: pred for eid, pred in zip(full_dataset.essay_ids, all_preds)}
id_to_score = {
    eid: score for eid, score in zip(full_dataset.essay_ids, full_dataset.scores)
}
id_to_idx = {eid: idx for idx, eid in enumerate(full_dataset.essay_ids)}

# Extract validation subset
for eid in val_meta_df["essay_id"].values:
    if eid in id_to_pred:
        val_preds.append(id_to_pred[eid])
        val_trues.append(id_to_score[eid])
        val_indices.append(id_to_idx[eid])

val_preds = np.array(val_preds)
val_trues = np.array(val_trues)

# Calculate Metric
val_qwk = utils.compute_qwk(val_trues, val_preds)
print(f"Final Validation Metric: {val_qwk}")

# ==============================================================================
# 4. Failure Analysis
# ==============================================================================
print("\nPerforming Failure Analysis...")
# Calculate error magnitude (absolute difference)
errors = np.abs(val_preds - val_trues)

# Get meta-features for the validation subset
val_meta_features = full_dataset.meta_features[val_indices]

print("Correlation between Error Magnitude and Meta-Features:")
for i, feature_name in enumerate(Config.meta_features):
    feature_values = val_meta_features[:, i]
    # Handle potential constant values to avoid RuntimeWarnings
    if np.std(feature_values) > 0 and np.std(errors) > 0:
        corr = np.corrcoef(errors, feature_values)[0, 1]
        print(f"{feature_name}: {corr:.4f}")
    else:
        print(f"{feature_name}: 0.0000 (Constant values)")

# ==============================================================================
# 5. Submission Generation
# ==============================================================================
TARGET_METRIC = 0.8246384329994252

if val_qwk > TARGET_METRIC:
    print(
        f"\nValidation metric ({val_qwk}) meets threshold ({TARGET_METRIC}). Generating submission..."
    )
    pipeline.generate_submission()
else:
    print(
        f"\nValidation metric ({val_qwk}) does not meet threshold ({TARGET_METRIC}). Submission skipped."
    )
