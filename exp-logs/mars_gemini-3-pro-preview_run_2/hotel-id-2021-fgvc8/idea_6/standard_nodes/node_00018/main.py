import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.engine import train_model, get_model_embeddings
from library.inference import concat_features, apply_dba, apply_qe, predict


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    Config.EPOCHS = 3
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 4

    # Clean up potentially stale embeddings from previous runs
    for split in ["train", "val", "test"]:
        for model in Config.MODEL_NAMES:
            p_emb = os.path.join(Config.WORKING_DIR, f"{split}_embeddings_{model}.npy")
            p_lbl = os.path.join(Config.WORKING_DIR, f"{split}_labels_{model}.npy")
            if os.path.exists(p_emb):
                os.remove(p_emb)
            if os.path.exists(p_lbl):
                os.remove(p_lbl)

    print("=== Starting Fast Baseline Run ===")

    # 2. Data Preparation (Sampling)
    # We need to sample training data but ensure all classes are present for ArcFace
    original_train_path = Config.TRAIN_METADATA_PATH
    df_train = pd.read_csv(original_train_path)

    print(
        f"Original Training Data: {len(df_train)} images, {df_train['hotel_id'].nunique()} classes"
    )

    # Strategy: Take 1 image per class to ensure coverage, then sample remaining
    df_keep = df_train.groupby("hotel_id").head(1)
    df_rest = df_train.drop(df_keep.index)

    # Target size ~30k images for speed
    target_size = 30000
    remaining_needed = target_size - len(df_keep)

    if remaining_needed > 0 and len(df_rest) > 0:
        df_sample = df_rest.sample(
            n=min(remaining_needed, len(df_rest)), random_state=Config.SEED
        )
        df_final = pd.concat([df_keep, df_sample])
    else:
        df_final = df_keep

    # Shuffle
    df_final = df_final.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

    print(f"Sampled Training Data: {len(df_final)} images")

    # Save sampled metadata
    sampled_train_path = os.path.join(Config.WORKING_DIR, "train_metadata_sampled.csv")
    df_final.to_csv(sampled_train_path, index=False)

    # Update Config to point to sampled data
    Config.TRAIN_METADATA_PATH = sampled_train_path

    # Remove cached mapping to force regeneration based on sampled data
    mapping_path = os.path.join(Config.WORKING_DIR, "class_mapping.parquet")
    if os.path.exists(mapping_path):
        os.remove(mapping_path)

    # 3. Training
    device = Config.DEVICE
    # Get dataloaders (this will generate the mapping based on df_final)
    # load_cached_data=False ensures we rebuild mapping from our new sampled file
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = get_dataloaders(
        load_cached_data=False
    )

    for model_name in Config.MODEL_NAMES:
        # Train model
        # Passing epochs explicitly to override default in function signature
        train_model(model_name, train_loader, val_loader, device, epochs=Config.EPOCHS)

    # 4. Validation & Failure Analysis
    print("\n=== Performing Validation & Failure Analysis ===")

    # Extract Embeddings for Validation Protocol
    # Gallery: Train Set (Sampled)
    # Query: Validation Set

    gallery_embs_list = []
    query_embs_list = []
    val_targets = []
    gallery_labels = None

    for model_name in Config.MODEL_NAMES:
        # Train (Gallery)
        g_emb, g_lbl = get_model_embeddings(
            model_name, "train", train_loader, device, load_cached_data=True
        )
        gallery_embs_list.append(g_emb)
        if gallery_labels is None:
            gallery_labels = g_lbl

        # Val (Query)
        v_emb, v_lbl = get_model_embeddings(
            model_name, "val", val_loader, device, load_cached_data=True
        )
        query_embs_list.append(v_emb)

        if len(val_targets) == 0:
            val_targets = v_lbl

    # Concat Features
    gallery_embeddings, query_embeddings = concat_features(
        gallery_embs_list, query_embs_list
    )

    # Database Augmentation (DBA)
    if Config.DBA_K > 0:
        gallery_embeddings = apply_dba(gallery_embeddings, k=Config.DBA_K)

    # Query Expansion (QE)
    if Config.QE_K > 0:
        query_embeddings = apply_qe(query_embeddings, gallery_embeddings, k=Config.QE_K)

    # Retrieval
    print("Calculating Validation Metrics...")
    knn = NearestNeighbors(n_neighbors=Config.TOP_K, metric="cosine", n_jobs=-1)
    knn.fit(gallery_embeddings)
    _, indices = knn.kneighbors(query_embeddings)

    # Calculate MAP and Per-Instance AP for Failure Analysis
    aps = []
    for i in range(len(query_embeddings)):
        target = val_targets[i]
        neighbor_idxs = indices[i]
        pred_labels = gallery_labels[neighbor_idxs]

        # AP@5
        ap = 0.0
        if target in pred_labels:
            # rank is 1-based
            rank = np.where(pred_labels == target)[0][0] + 1
            ap = 1.0 / rank
        aps.append(ap)

    final_val_map = np.mean(aps)
    print(f"Final Validation Metric: {final_val_map:.16f}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Ensure alignment
    min_len = min(len(val_df), len(aps))
    val_df = val_df.iloc[:min_len].copy()
    aps = aps[:min_len]

    val_df["ap"] = aps
    val_df["error"] = 1.0 - val_df["ap"]

    # Feature Engineering
    val_df["chain_id"] = val_df["chain"]

    if "timestamp" in val_df.columns:
        val_df["dt"] = pd.to_datetime(val_df["timestamp"], errors="coerce")
        val_df["year"] = val_df["dt"].dt.year.fillna(0)
        val_df["month"] = val_df["dt"].dt.month.fillna(0)
        val_df["hour"] = val_df["dt"].dt.hour.fillna(0)
    else:
        val_df["year"] = 0
        val_df["month"] = 0
        val_df["hour"] = 0

    # Calculate Correlations
    features = ["chain_id", "year", "month", "hour"]
    correlations = val_df[features + ["error"]].corr()["error"].drop("error")

    print("Correlation between Error and Input Features:")
    print(correlations)

    # 5. Submission
    threshold = 0.7120973100214514
    if final_val_map > threshold:
        print(
            f"\nValidation Metric {final_val_map} > {threshold}. Generating Submission..."
        )

        # Reuse Gallery (Train) from Validation step
        # Extract Test Embeddings (Query)
        test_query_embs_list = []
        test_ids = None

        for model_name in Config.MODEL_NAMES:
            t_emb, t_id = get_model_embeddings(
                model_name, "test", test_loader, device, load_cached_data=True
            )
            test_query_embs_list.append(t_emb)
            if test_ids is None:
                test_ids = t_id

        # Concat
        # Note: concat_features normalizes inputs. We pass the raw lists again to be consistent.
        gallery_final, test_query_final = concat_features(
            gallery_embs_list, test_query_embs_list
        )

        # DBA
        if Config.DBA_K > 0:
            gallery_final = apply_dba(gallery_final, k=Config.DBA_K)

        # QE
        if Config.QE_K > 0:
            test_query_final = apply_qe(test_query_final, gallery_final, k=Config.QE_K)

        # Predict
        df_sub = predict(
            test_query_final,
            gallery_final,
            gallery_labels,
            test_ids,
            idx_to_class,
            top_k=Config.TOP_K,
        )

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric {final_val_map} <= {threshold}. Skipping Submission."
        )


if __name__ == "__main__":
    main()
