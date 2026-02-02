import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, weighted_log_loss, load_checkpoint
from library.engine import (
    train_segmentor,
    train_encoder,
    extract_features,
    train_aggregator,
)
from library.models import AnatomicalTransformer
from library.inference import InferencePipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def prepare_consistent_inference_cache():
    """
    Creates a cache file for InferencePipeline that is consistent with the training data.
    Since training used empty anatomical IDs (zeros), we must force inference to use zeros
    instead of the real IDs extracted by InferencePipeline, otherwise the embeddings will be OOD.
    Also avoids re-running feature extraction.
    """
    print("Preparing consistent inference cache...")
    if not os.path.exists(Config.TEST_FEATURES_CACHE):
        print("Test features not found. Cannot prepare cache.")
        return

    feats_dict = np.load(Config.TEST_FEATURES_CACHE, allow_pickle=True).item()
    inference_data = {}

    for uid, feats in feats_dict.items():
        # feats is (Seq_Len, Feature_Dim)
        seq_len = feats.shape[0]
        # Force anat_ids to 0 to match training conditions
        anat_ids = np.zeros(seq_len, dtype=np.int64)

        inference_data[uid] = {"features": feats, "anat_ids": anat_ids}

    # Save to the location InferencePipeline expects
    # InferencePipeline looks for 'test_inference_data.npy' in CACHE_DIR
    save_path = os.path.join(Config.CACHE_DIR, "test_inference_data.npy")
    np.save(save_path, inference_data)
    print(f"Saved consistent inference data to {save_path}")


def run_validation_and_analysis():
    print("\n" + "=" * 40)
    print("Final Validation & Failure Analysis")
    print("=" * 40)

    # 1. Load Model
    model = AnatomicalTransformer().to(Config.DEVICE)
    load_checkpoint(model, None, Config.AGG_MODEL_PATH, device=Config.DEVICE)
    model.eval()

    # 2. Load Validation Data
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    if not os.path.exists(Config.VAL_FEATURES_CACHE):
        print("Validation features not found.")
        return 1.0  # Return high loss

    val_feats_dict = np.load(Config.VAL_FEATURES_CACHE, allow_pickle=True).item()

    preds_all = []
    targets_all = []
    uids = []

    target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # 3. Inference Loop
    with torch.no_grad():
        for _, row in val_df.iterrows():
            uid = row["StudyInstanceUID"]
            uids.append(uid)

            # Ground Truth
            y_true = row[target_cols].values.astype(np.float32)
            targets_all.append(y_true)

            # Features
            if uid in val_feats_dict:
                feats = val_feats_dict[uid]
            else:
                feats = np.zeros((10, Config.ENC_FEATURE_DIM), dtype=np.float32)

            # Preprocessing (Pad/Truncate) - matching Dataset logic
            seq_len = feats.shape[0]
            max_len = Config.AGG_MAX_SEQ_LEN

            if seq_len > max_len:
                start = (seq_len - max_len) // 2
                feats = feats[start : start + max_len]
                mask = np.ones(max_len, dtype=np.float32)
            else:
                pad_len = max_len - seq_len
                feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="constant")
                mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)]).astype(
                    np.float32
                )

            # Anatomical IDs (Zeros for consistency with training)
            anat_ids = np.zeros(max_len, dtype=np.int64)

            # Tensorize
            feats_t = torch.from_numpy(feats).float().unsqueeze(0).to(Config.DEVICE)
            anat_ids_t = (
                torch.from_numpy(anat_ids).long().unsqueeze(0).to(Config.DEVICE)
            )
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(Config.DEVICE)

            # Predict
            logits = model(feats_t, anat_ids_t, mask_t)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds_all.append(probs)

    preds_all = np.array(preds_all)
    targets_all = np.array(targets_all)

    # 4. Compute Final Metric
    final_metric = weighted_log_loss(targets_all, preds_all)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Compute per-sample weighted log loss
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
    epsilon = 1e-15
    p = np.clip(preds_all, epsilon, 1 - epsilon)
    loss_matrix = -(targets_all * np.log(p) + (1 - targets_all) * np.log(1 - p))
    weighted_loss = (loss_matrix * weights).mean(axis=1)  # (N,)

    # Metadata for correlation
    # Num Slices
    num_slices = []
    for idx, row in val_df.iterrows():
        img_dir = os.path.join(Config.INPUT_DIR, row["image_path"])
        try:
            # Fast count
            n = len([name for name in os.listdir(img_dir) if name.endswith(".dcm")])
        except:
            n = 0
        num_slices.append(n)

    # Fracture Count (Ground Truth)
    fracture_counts = targets_all[:, :7].sum(axis=1)

    # Create DataFrame
    df_fail = pd.DataFrame(
        {
            "loss": weighted_loss,
            "num_slices": num_slices,
            "fracture_count": fracture_counts,
        }
    )

    # Correlations
    corr_slices = df_fail["loss"].corr(df_fail["num_slices"])
    corr_frac = df_fail["loss"].corr(df_fail["fracture_count"])

    print("\nFailure Analysis Correlations (Error Magnitude vs Feature):")
    print(f"  Num Slices: {corr_slices:.4f}")
    print(f"  Fracture Count: {corr_frac:.4f}")

    return final_metric


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Training Pipeline
    # Stage 1: Segmentation
    train_segmentor(load_cached_data=True)

    # Stage 2: Encoder
    train_encoder(load_cached_data=True)

    # Feature Extraction (Train/Val/Test)
    extract_features(load_cached_data=True)

    # Stage 3: Aggregator
    train_aggregator(load_cached_data=True)

    # 3. Validation & Analysis
    metric = run_validation_and_analysis()

    # 4. Conditional Submission
    THRESHOLD = 0.9254394427010018

    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric:.6f}) is lower than threshold ({THRESHOLD}). Generating Submission..."
        )

        # Prepare consistent cache to avoid OOD embeddings and re-extraction
        prepare_consistent_inference_cache()

        # Run Inference Pipeline
        pipeline = InferencePipeline()
        pipeline.predict(load_cached_data=True)

    else:
        print(
            f"\nMetric ({metric:.6f}) is NOT lower than threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
