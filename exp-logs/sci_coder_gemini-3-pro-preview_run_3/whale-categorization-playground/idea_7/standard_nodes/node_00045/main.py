import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import from provided library files
from library.config import Config
from library.utils import set_seed, calculate_map5
from library.dataset import get_dataloaders, WhaleDataset, get_transforms
from library.trainer import train_model
from library.inference import extract_embeddings
from library.post_process import k_reciprocal_rerank, query_expansion


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Set seeds for reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Override Config for optimized runtime within limits
    # Extended training for convergence (Cite solution_lesson_node_00019)
    Config.EPOCHS = 24
    Config.BATCH_SIZE = 16

    print("=== Configuration ===")
    print(f"Device: {device}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Backbones: {Config.MODEL_BACKBONES}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n=== Data Loading ===")
    # Load standard dataloaders (Train is shuffled/augmented, Val/Test are standard)
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        load_cached_data=True
    )

    # Create a specific 'Gallery' loader.
    # The Gallery consists of the training data but processed deterministically (no shuffle, val transforms)
    # for inference/retrieval purposes.
    df_train = pd.read_csv(Config.TRAIN_CSV)
    # Filter out new_whale to match the training set logic (Gallery = Known Whales)
    df_train = df_train[df_train["Id"] != "new_whale"].reset_index(drop=True)

    # Load the class mapping to ensure indices match
    classes_path = os.path.join(Config.WORKING_DIR, "label_encoder_classes.npy")
    if os.path.exists(classes_path):
        unique_classes = np.load(classes_path, allow_pickle=True)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        df_train["label_idx"] = df_train["Id"].map(class_to_idx)
    else:
        raise FileNotFoundError(
            "Label encoder not found. Ensure get_dataloaders runs first."
        )

    gallery_dataset = WhaleDataset(
        df_train,
        mode="val",  # Use validation transforms (Resize + Normalize)
        transforms=get_transforms(mode="val"),
    )

    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(f"Gallery Loader created with {len(df_train)} samples.")

    # -------------------------------------------------------------------------
    # 3. Model Training & Feature Extraction
    # -------------------------------------------------------------------------
    models_to_train = Config.MODEL_BACKBONES

    # Storage for fused embeddings
    gallery_feats_list = []
    val_feats_list = []
    test_feats_list = []

    # Targets/Names (Shared across models)
    gallery_targets = None
    val_targets = None
    test_filenames = None

    for model_name in models_to_train:
        print(f"\n--- Processing Model: {model_name} ---")

        # A. Train
        model = train_model(model_name, train_loader, val_loader, num_classes, device)

        # B. Extract Embeddings
        print(f"Extracting embeddings for {model_name}...")

        # Gallery (Train Set)
        g_emb, g_tgt = extract_embeddings(model, gallery_loader, device)
        gallery_feats_list.append(g_emb)
        if gallery_targets is None:
            gallery_targets = g_tgt

        # Query (Validation Set)
        v_emb, v_tgt = extract_embeddings(model, val_loader, device)
        val_feats_list.append(v_emb)
        if val_targets is None:
            val_targets = v_tgt

        # Test Set
        t_emb, t_names = extract_embeddings(model, test_loader, device)
        test_feats_list.append(t_emb)
        if test_filenames is None:
            test_filenames = t_names

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Feature Fusion
    # -------------------------------------------------------------------------
    print("\n=== Feature Fusion ===")
    # Concatenate embeddings from all models: Shape (N, D1 + D2)
    gallery_feats = np.concatenate(gallery_feats_list, axis=1)
    val_feats = np.concatenate(val_feats_list, axis=1)
    test_feats = np.concatenate(test_feats_list, axis=1)

    # Convert to Tensor and Normalize
    gallery_feats_t = torch.from_numpy(gallery_feats).to(device)
    val_feats_t = torch.from_numpy(val_feats).to(device)
    test_feats_t = torch.from_numpy(test_feats).to(device)

    gallery_feats_t = F.normalize(gallery_feats_t, p=2, dim=1)
    val_feats_t = F.normalize(val_feats_t, p=2, dim=1)
    test_feats_t = F.normalize(test_feats_t, p=2, dim=1)

    # -------------------------------------------------------------------------
    # 5. Validation & Metrics
    # -------------------------------------------------------------------------
    print("\n=== Validation ===")

    # Apply Query Expansion (Cite solution_lesson_node_00032)
    print("Applying Query Expansion...")
    val_feats_t = query_expansion(val_feats_t, gallery_feats_t, top_k=5)
    test_feats_t = query_expansion(test_feats_t, gallery_feats_t, top_k=5)

    # Compute distance matrix using k-Reciprocal Re-ranking
    # This acts as the retrieval step: Val Queries vs Train Gallery
    dist_mat_val = k_reciprocal_rerank(
        val_feats_t, gallery_feats_t, k1=20, k2=6, lambda_value=0.3
    )
    dist_mat_val = dist_mat_val.cpu().numpy()

    # Calculate MAP@5
    # Get top 5 indices for each query
    # argsort sorts ascending (smallest distance first)
    top_k_indices_val = np.argsort(dist_mat_val, axis=1)[:, :5]
    top_k_dists_val = np.sort(dist_mat_val, axis=1)[:, :5]

    # Cite debug_lesson_3: Mirror Open-Set Inference Logic in Validation Pipelines
    # Cite debug_lesson_5: Sanitize Sentinel Labels Before Inverse Transformation
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    NEW_WHALE_THRESHOLD = 0.6

    val_preds_str = []
    val_targets_str = []

    for i in range(len(val_targets)):
        # 1. Process Target (Handle -1 -> new_whale)
        tgt_idx = val_targets[i]
        if tgt_idx == -1:
            val_targets_str.append("new_whale")
        else:
            val_targets_str.append(idx_to_class[tgt_idx])

        # 2. Process Prediction
        indices = top_k_indices_val[i]
        dists = top_k_dists_val[i]

        current_preds = []

        # Logic: If the best match is too far, predict new_whale first.
        if dists[0] > NEW_WHALE_THRESHOLD:
            current_preds.append("new_whale")

        # Add retrieved labels
        for idx in indices:
            label = idx_to_class[gallery_targets[idx]]
            if len(current_preds) < 5:
                current_preds.append(label)

        # If we haven't filled 5 slots and haven't added new_whale yet, append it
        if "new_whale" not in current_preds and len(current_preds) < 5:
            current_preds.append("new_whale")

        val_preds_str.append(current_preds[:5])

    val_map5 = calculate_map5(val_preds_str, val_targets_str)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_map5:.16f}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")
    # Determine correctness (Rank-1 accuracy)
    is_correct = []
    for i in range(len(val_targets)):
        if val_predictions[i][0] == val_targets[i]:
            is_correct.append(1)
        else:
            is_correct.append(0)

    errors = 1 - np.array(is_correct)

    # Get Validation Image File Sizes
    # We need to read the validation metadata again to get paths
    df_val = pd.read_csv(Config.VAL_CSV)
    df_val = df_val[df_val["Id"] != "new_whale"].reset_index(drop=True)

    file_sizes = []
    for p in df_val["file_path"]:
        full_path = os.path.join(Config.INPUT_DIR, p)
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
        else:
            file_sizes.append(0)

    # Calculate Correlation
    if len(file_sizes) == len(errors):
        # Use numpy for correlation
        if np.std(errors) > 0 and np.std(file_sizes) > 0:
            corr = np.corrcoef(errors, file_sizes)[0, 1]
            print(f"Correlation between Error and File Size: {corr:.4f}")
        else:
            print("Correlation undefined (variance is zero).")
    else:
        print("Warning: Mismatch in validation set size for analysis.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD_METRIC = 0.9019736842105263

    if val_map5 > THRESHOLD_METRIC:
        print("\n=== Generating Submission ===")

        # Compute Distance Matrix (Test vs Gallery)
        dist_mat_test = k_reciprocal_rerank(
            test_feats_t, gallery_feats_t, k1=20, k2=6, lambda_value=0.3
        )
        dist_mat_test = dist_mat_test.cpu().numpy()

        # Prepare submission data
        submission_data = []

        # Inverse mapping: label_idx -> class_name
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Get top 5 candidates
        top_k_indices_test = np.argsort(dist_mat_test, axis=1)[:, :5]
        top_k_dists_test = np.sort(dist_mat_test, axis=1)[:, :5]

        # Threshold for 'new_whale' prediction
        # Jaccard distance is between 0 and 1.
        # A threshold of 0.6 is generally robust for ReID tasks.
        NEW_WHALE_THRESHOLD = 0.6

        for i in range(len(test_filenames)):
            fname = test_filenames[i]
            indices = top_k_indices_test[i]
            dists = top_k_dists_test[i]

            # Map indices to class names
            pred_labels = [idx_to_class[gallery_targets[idx]] for idx in indices]

            final_preds = []

            # Logic: If the best match is too far, predict new_whale first.
            if dists[0] > NEW_WHALE_THRESHOLD:
                final_preds.append("new_whale")

            # Add retrieved labels
            for label in pred_labels:
                if len(final_preds) < 5:
                    final_preds.append(label)

            # If we haven't filled 5 slots and haven't added new_whale yet, append it at the end
            if "new_whale" not in final_preds and len(final_preds) < 5:
                final_preds.append("new_whale")

            # Ensure exactly 5 predictions
            final_preds = final_preds[:5]

            submission_data.append([fname, " ".join(final_preds)])

        # Save Submission
        df_sub = pd.DataFrame(submission_data, columns=["Image", "Id"])
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation Metric ({val_map5:.6f}) did not exceed threshold ({THRESHOLD_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
