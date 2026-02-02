import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
import importlib

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.trainer as trainer_lib
import library.inference as inference_lib

# Cite debug_lesson_1: Python Module Caching Invalidates Runtime Code Modifications
# We must explicitly reload the modules to ensure the new config values (Batch Size 16, Image Size 384)
# are picked up by the persistent Python process.
importlib.reload(config)
importlib.reload(utils)
importlib.reload(data_loader)
importlib.reload(model_lib)
importlib.reload(trainer_lib)
importlib.reload(inference_lib)


def calculate_correlation(x, y):
    """
    Helper to calculate Pearson correlation coefficient between two arrays.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    # np.corrcoef returns a covariance matrix, [0,1] is the correlation
    return np.corrcoef(x, y)[0, 1]


def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    utils.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    # get_dataloaders filters out 'new_whale' for train/val to suit ArcFace training
    train_loader, val_loader, gallery_loader, test_loader, label_map, num_classes = (
        data_loader.get_dataloaders()
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print(f"Initializing WhaleEfficientNet for {num_classes} classes...")
    model = model_lib.WhaleEfficientNet(num_classes=num_classes)
    model = model.to(device)

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------
    # Using config.NUM_EPOCHS (30) to allow full convergence
    TRAIN_EPOCHS = config.NUM_EPOCHS
    print(f"Starting training for {TRAIN_EPOCHS} epochs...")

    trainer = trainer_lib.Trainer(
        model, train_loader, val_loader, gallery_loader, num_classes
    )
    trainer.fit(num_epochs=TRAIN_EPOCHS)

    # -------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Final Validation and Failure Analysis...")

    # Load the best model weights saved by the trainer
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    else:
        print("Warning: Model checkpoint not found. Using current weights.")

    model.eval()

    # A. Extract Gallery Embeddings (Reference)
    gallery_embeddings = []
    gallery_labels_str = []

    with torch.no_grad():
        for images, _, label_strs in gallery_loader:
            images = images.to(device)
            embeds = model(images)
            gallery_embeddings.append(embeds.cpu())
            gallery_labels_str.extend(label_strs)

    gallery_embeddings = torch.cat(gallery_embeddings)
    gallery_labels_str = np.array(gallery_labels_str)

    # B. Extract Query Embeddings (Validation)
    query_embeddings = []
    query_labels_str = []
    query_filenames = []

    # Access the dataset to get filenames corresponding to the loader batches
    val_dataset = val_loader.dataset

    with torch.no_grad():
        for i, (images, _, label_strs) in enumerate(val_loader):
            images = images.to(device)
            embeds = model(images)
            query_embeddings.append(embeds.cpu())
            query_labels_str.extend(label_strs)

            # Retrieve filenames
            start_idx = i * config.BATCH_SIZE
            end_idx = start_idx + images.size(0)
            end_idx = min(end_idx, len(val_dataset.filenames))
            batch_filenames = val_dataset.filenames[start_idx:end_idx]
            query_filenames.extend(batch_filenames)

    query_embeddings = torch.cat(query_embeddings)

    # C. Compute Similarity & Metrics
    gal_emb_cuda = gallery_embeddings.to(device)
    qry_emb_cuda = query_embeddings.to(device)

    sim_matrix = torch.matmul(qry_emb_cuda, gal_emb_cuda.t())

    # Get Top 50 for filtering
    top_vals, top_inds = torch.topk(sim_matrix, k=50, dim=1)
    top_vals = top_vals.cpu().numpy()
    top_inds = top_inds.cpu().numpy()

    predictions = []
    targets = query_labels_str

    # Data for failure analysis
    is_error_list = []
    confidence_list = []

    threshold = config.CONFIDENCE_THRESHOLD

    for i in range(len(targets)):
        true_label = targets[i]
        scores = top_vals[i]
        indices = top_inds[i]

        candidate_ids = gallery_labels_str[indices]

        unique_ids = []
        seen = set()
        for cid in candidate_ids:
            if cid not in seen:
                unique_ids.append(cid)
                seen.add(cid)
            if len(unique_ids) >= 5:
                break

        best_score = scores[0]
        if best_score < threshold:
            pred_labels = ["new_whale"] + unique_ids[:4]
            primary_pred = "new_whale"
        else:
            pred_labels = [unique_ids[0], "new_whale"] + unique_ids[1:4]
            primary_pred = unique_ids[0]

        predictions.append(pred_labels)

        # Check Top-1 Accuracy
        is_error = 1 if primary_pred != true_label else 0
        is_error_list.append(is_error)
        confidence_list.append(best_score)

    # Calculate Final Metric
    final_metric = utils.map5(predictions, targets)
    print(f"Final Validation Metric: {final_metric}")

    # D. Failure Analysis (Correlations)
    # Load validation metadata to get file paths for image property analysis
    df_val_meta = pd.read_csv(config.VAL_CSV)

    # Create a dataframe of results
    df_results = pd.DataFrame(
        {
            "Image": query_filenames,
            "is_error": is_error_list,
            "confidence": confidence_list,
        }
    )

    # Merge to associate results with file paths
    # Note: val_loader filtered out 'new_whale', so df_results is a subset of df_val_meta
    df_merged = pd.merge(df_val_meta, df_results, on="Image", how="inner")

    # Calculate image properties (Area, Aspect Ratio)
    widths = []
    heights = []
    aspect_ratios = []

    for idx, row in df_merged.iterrows():
        fpath = os.path.join(config.INPUT_DIR, row["file_path"])
        # Read image dimensions
        img = cv2.imread(fpath)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
        else:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    df_merged["area"] = np.array(widths) * np.array(heights)
    df_merged["aspect_ratio"] = np.array(aspect_ratios)

    # Calculate Correlations
    corr_conf = calculate_correlation(df_merged["is_error"], df_merged["confidence"])
    corr_area = calculate_correlation(df_merged["is_error"], df_merged["area"])
    corr_ar = calculate_correlation(df_merged["is_error"], df_merged["aspect_ratio"])

    print("\n--- Failure Analysis (Correlation with Error Magnitude) ---")
    print(
        f"Confidence:   {corr_conf:.4f} (Expected negative: higher confidence -> lower error)"
    )
    print(f"Image Area:   {corr_area:.4f}")
    print(f"Aspect Ratio: {corr_ar:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.8543859649122806

    if final_metric > THRESHOLD:
        print(f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission...")
        # run_inference handles loading the model and generating the CSV
        inference_lib.run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
