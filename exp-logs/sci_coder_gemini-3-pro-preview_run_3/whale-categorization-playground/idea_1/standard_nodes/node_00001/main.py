import os
import numpy as np
import pandas as pd
import torch
import cv2

# Import library modules
from library.config import Config
from library.engine import train_model, predict_submission, generate_embeddings
from library.dataset import WhaleInferenceDataset, get_transforms
from library.utils import seed_everything
from torch.utils.data import DataLoader


def calculate_map5(val_df, predictions):
    """
    Computes MAP@5 score.
    val_df: DataFrame containing 'Id' column with ground truth.
    predictions: List of lists, where each inner list contains up to 5 predicted labels.
    """
    scores = []
    for idx, row in val_df.iterrows():
        ground_truth = row["Id"]
        preds = predictions[idx]

        score = 0.0
        for rank, pred_label in enumerate(preds):
            if pred_label == ground_truth:
                score = 1.0 / (rank + 1)
                break
        scores.append(score)

    return np.mean(scores)


def analyze_failures(val_df, predictions, min_distances):
    """
    Performs failure analysis on the validation set.
    """
    print("\n--- Failure Analysis ---")

    # 1. Calculate correctness (Top-1)
    is_correct_list = []
    for idx, row in val_df.iterrows():
        ground_truth = row["Id"]
        # Top-1 prediction
        pred_label = predictions[idx][0]
        is_correct_list.append(1 if pred_label == ground_truth else 0)

    is_correct = np.array(is_correct_list)
    min_dists = np.array(min_distances)

    # 2. Correlation with Model Confidence (Distance)
    # Pearson correlation between binary (0/1) and continuous is equivalent to Point-Biserial
    if len(np.unique(is_correct)) > 1:
        corr_matrix = np.corrcoef(is_correct, min_dists)
        corr_dist = corr_matrix[0, 1]
        print(f"Correlation (Correctness vs. Min Distance): {corr_dist}")
        print(
            "  (Negative correlation implies lower distance (higher confidence) leads to better accuracy)"
        )
    else:
        print(
            "Correlation (Correctness vs. Min Distance): Undefined (all samples correct or all wrong)"
        )

    # 3. Correlation with Image Brightness
    print("Computing image stats for correlation analysis...")
    brightness_list = []

    for idx, row in val_df.iterrows():
        fpath = os.path.join(Config.INPUT_DIR, row["file_path"])
        # Read directly with cv2 for speed
        img = cv2.imread(fpath)
        if img is not None:
            # simple mean brightness
            mean_val = img.mean()
            brightness_list.append(mean_val)
        else:
            brightness_list.append(0.0)

    brightness = np.array(brightness_list)

    if len(np.unique(is_correct)) > 1:
        corr_matrix_bright = np.corrcoef(is_correct, brightness)
        corr_bright = corr_matrix_bright[0, 1]
        print(f"Correlation (Correctness vs. Image Brightness): {corr_bright}")
    else:
        print("Correlation (Correctness vs. Image Brightness): Undefined")


def run_validation(model, device):
    """
    Runs inference on the validation set and computes MAP@5.
    """
    print("Generating Validation Embeddings...")

    # 1. Gallery (Train Set - No new_whale)
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_gallery = df_train[df_train["Id"] != "new_whale"].reset_index(drop=True)

    # Create temp csv for gallery dataset
    gallery_csv = os.path.join(Config.WORKING_DIR, "val_gallery.csv")
    df_gallery.to_csv(gallery_csv, index=False)

    gallery_dataset = WhaleInferenceDataset(
        csv_file=gallery_csv, transform=get_transforms(mode="test")
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    gallery_embeddings, _, gallery_ids = generate_embeddings(
        model, gallery_loader, device
    )
    gallery_ids = np.array(gallery_ids)

    # 2. Query (Validation Set)
    val_dataset = WhaleInferenceDataset(
        csv_file=Config.VAL_CSV, transform=get_transforms(mode="test")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    query_embeddings, _, query_ids = generate_embeddings(model, val_loader, device)

    # 3. Compute Distances
    print("Computing distances...")
    q_tensor = torch.from_numpy(query_embeddings).to(device)
    g_tensor = torch.from_numpy(gallery_embeddings).to(device)

    # Pairwise distance
    dists = torch.cdist(q_tensor, g_tensor, p=2)

    # 4. Top K & Prediction
    k = min(Config.KNN_K, len(gallery_ids))
    topk_vals, topk_indices = torch.topk(dists, k=k, dim=1, largest=False)

    topk_vals = topk_vals.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()

    predictions = []
    min_distances = []

    for i in range(len(query_ids)):
        neighbor_indices = topk_indices[i]
        neighbor_dists = topk_vals[i]

        neighbor_ids = gallery_ids[neighbor_indices]

        # Unique IDs logic
        unique_ids = []
        seen = set()
        for nid in neighbor_ids:
            if nid not in seen:
                unique_ids.append(nid)
                seen.add(nid)
            if len(unique_ids) >= 5:
                break

        nearest_dist = neighbor_dists[0]
        min_distances.append(nearest_dist)

        final_preds = []
        if nearest_dist > Config.NEW_WHALE_THRESHOLD:
            final_preds.append("new_whale")
            final_preds.extend(unique_ids)
        else:
            final_preds.extend(unique_ids)
            final_preds.append("new_whale")

        # Truncate/Clean to top 5 unique
        clean_preds = []
        seen_final = set()
        for p in final_preds:
            if p not in seen_final:
                clean_preds.append(p)
                seen_final.add(p)
            if len(clean_preds) == 5:
                break

        predictions.append(clean_preds)

    # 5. Calculate Metric
    df_val = pd.read_csv(Config.VAL_CSV)
    metric = calculate_map5(df_val, predictions)

    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    analyze_failures(df_val, predictions, min_distances)

    # Cleanup
    if os.path.exists(gallery_csv):
        os.remove(gallery_csv)


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    # Fast Baseline Config Overrides
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 128

    print("Starting Fast Baseline Training...")

    # 2. Train
    model = train_model(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, subset_size=None
    )

    # 3. Validation & Analysis
    print("\nStarting Validation...")
    run_validation(model, Config.DEVICE)

    # 4. Submission
    print("\nGenerating Submission...")
    predict_submission(
        model_path=Config.MODEL_PATH,
        batch_size=Config.BATCH_SIZE,
        subset_size=None,
        load_cached_data=False,
    )

    print("Done.")


if __name__ == "__main__":
    main()
