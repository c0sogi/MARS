import os
import cv2
import torch
import numpy as np
import pandas as pd
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, map_at_5
from library.dataset import get_dataloaders
from library.model import WhaleModel
from library.engine import WhaleEngine
from library.inference import generate_predictions
from library.rerank import re_ranking

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Data Loading
    # load_cached_data=True will use existing .npy files in ./working/idea_4/ if available
    train_loader, val_loader, test_loader, label_encoder, num_classes = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    model = WhaleModel(num_classes=num_classes)

    # 4. Engine Initialization
    engine = WhaleEngine(model, train_loader, val_loader, test_loader, label_encoder)

    # 5. Training
    # Fits the model and saves the best version to Config.MODEL_SAVE_PATH
    engine.fit()

    # 6. Final Validation & Failure Analysis
    print("\nRunning Final Validation and Failure Analysis...")

    # Load the best model weights
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Best model file not found.")
        return

    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE)
    model.eval()

    # Extract features for Validation set (Query) and Training set (Gallery)
    # Note: engine.gallery_loader contains only the Training set (Known Whales)
    val_feats, val_labels = engine.extract_features(val_loader)
    gallery_feats, gallery_labels = engine.extract_features(engine.gallery_loader)

    val_feats = val_feats.to(Config.DEVICE)
    gallery_feats = gallery_feats.to(Config.DEVICE)

    # L2 Normalize
    val_feats = torch.nn.functional.normalize(val_feats, p=2, dim=1)
    gallery_feats = torch.nn.functional.normalize(gallery_feats, p=2, dim=1)

    # Compute Distance/Similarity Matrix
    if Config.USE_RERANKING:
        # Returns distance matrix (smaller is better)
        dist_matrix = re_ranking(
            val_feats,
            gallery_feats,
            k1=Config.RERANK_K1,
            k2=Config.RERANK_K2,
            lambda_value=Config.RERANK_LAMBDA,
        )
        sorted_indices = np.argsort(dist_matrix, axis=1)

        # We also need raw cosine similarity for the Open-Set Threshold check
        # (Re-ranking mixes Jaccard and Euclidean, so scale is different)
        sim_matrix = torch.mm(val_feats, gallery_feats.t())
        max_sim_vals, _ = torch.max(sim_matrix, dim=1)
        max_sim_vals = max_sim_vals.cpu().numpy()
    else:
        sim_matrix = torch.mm(val_feats, gallery_feats.t())
        max_sim_vals, _ = torch.max(sim_matrix, dim=1)
        max_sim_vals = max_sim_vals.cpu().numpy()
        sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True).cpu().numpy()

    # Generate Predictions and Calculate Metric
    val_preds = []
    val_gt = []
    sample_aps = []  # Average Precision per sample

    gallery_labels_np = gallery_labels.numpy()
    val_labels_np = val_labels.numpy()
    inverse_encoder = {v: k for k, v in label_encoder.items()}

    for i in range(len(val_labels)):
        # Ground Truth
        gt_idx = val_labels_np[i]
        gt_id = inverse_encoder.get(gt_idx, "new_whale")
        val_gt.append(gt_id)

        # Predictions
        top_indices = sorted_indices[i, :5]
        top_labels = gallery_labels_np[top_indices]
        top_ids = [inverse_encoder[lbl] for lbl in top_labels]

        # Open-Set Rejection Logic
        if max_sim_vals[i] < Config.NEW_WHALE_THRESHOLD:
            preds = ["new_whale"] + top_ids[:4]
        else:
            preds = top_ids

        val_preds.append(preds)

        # Calculate AP for this sample
        ap = 0.0
        if gt_id in preds:
            rank = preds.index(gt_id)  # 0-indexed
            ap = 1.0 / (rank + 1)
        sample_aps.append(ap)

    final_metric = map_at_5(val_preds, val_gt)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation with Input Features
    # Load validation metadata to get file paths
    df_val = pd.read_csv(Config.VAL_CSV)

    # Calculate Error Magnitude (1 - AP)
    # 0.0 means perfect prediction (Rank 1), 1.0 means failure (Not in top 5)
    errors = [1.0 - ap for ap in sample_aps]

    # Collect Image Statistics
    widths = []
    heights = []
    file_sizes = []

    # Iterate through validation set (assumes order is preserved, which it is for sequential loader)
    for idx, row in df_val.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if os.path.exists(path):
            file_sizes.append(os.path.getsize(path))
            # Read image dimensions
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "file_size": file_sizes}
    )

    # Compute Correlations
    corr_width = df_analysis["error"].corr(df_analysis["width"])
    corr_height = df_analysis["error"].corr(df_analysis["height"])
    corr_size = df_analysis["error"].corr(df_analysis["file_size"])

    print("Failure Analysis - Correlation with Error Magnitude:")
    print(f"  Width: {corr_width:.4f}")
    print(f"  Height: {corr_height:.4f}")
    print(f"  File Size: {corr_size:.4f}")

    # 7. Submission
    TARGET_SCORE = 0.751589
    if final_metric > TARGET_SCORE:
        print(
            f"Validation score {final_metric} exceeds target {TARGET_SCORE}. Generating submission..."
        )
        # Release memory before inference
        del model, engine, val_feats, gallery_feats, dist_matrix, sim_matrix
        torch.cuda.empty_cache()

        generate_predictions(load_cached_data=True)
    else:
        print(
            f"Validation score {final_metric} does not meet target {TARGET_SCORE}. Skipping submission."
        )


if __name__ == "__main__":
    main()
