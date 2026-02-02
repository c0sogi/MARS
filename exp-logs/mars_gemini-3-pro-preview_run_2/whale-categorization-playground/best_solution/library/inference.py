import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import CFG
from library.dataset import WhaleDataset, get_transforms, process_data
from library.modeling import WhaleModel
from library.engine import extract_embeddings
from library.utils import seed_everything


def get_embeddings(model, loader, device, tta=True):
    """
    Computes embeddings for a given dataset using a specific model.
    Applies Test-Time Augmentation (TTA) if enabled.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): The data loader.
        device (torch.device): The computation device.
        tta (bool): Whether to apply horizontal flip TTA.

    Returns:
        tuple: (embeddings (np.array), ids (list))
    """
    embeddings, ids = extract_embeddings(loader, model, device, tta=tta)
    return embeddings, ids


def generate_submission(
    model_a_path, model_b_path, output_path="./submission/submission.csv"
):
    """
    Generates the submission file using the Dual-Ensemble Strategy.

    1. Loads processed Train (Gallery) and Test (Query) data.
    2. Loads Model A (EfficientNet-B5) and Model B (EfficientNet-V2-M).
    3. Extracts embeddings for both datasets using both models with TTA.
    4. Computes Cosine Similarity matrices for both models.
    5. Fuses matrices (Average).
    6. Applies Open-Set Rejection (Thresholding) to handle 'new_whale'.
    7. Saves predictions to CSV.

    Args:
        model_a_path (str): Path to the checkpoint for Model A.
        model_b_path (str): Path to the checkpoint for Model B.
        output_path (str): Path to save the submission CSV.
    """
    seed_everything(CFG.seed)
    device = CFG.device

    print(f"Starting Inference Generation...")
    print(f"  Model A Path: {model_a_path}")
    print(f"  Model B Path: {model_b_path}")
    print(f"  Output Path:  {output_path}")

    # ---------------------------------------------------------
    # 1. Data Preparation
    # ---------------------------------------------------------
    # Load metadata (cached if available)
    train_df, _, test_df, num_classes = process_data(load_cached_data=True)

    # Define Transforms (Phase 2 Resolution: 384x384)
    transforms = get_transforms(data="test", image_size=CFG.image_size_p2)

    # Create Datasets
    # Gallery: Training Data. We need the 'Id' column to map predictions.
    gallery_dataset = WhaleDataset(train_df, transform=transforms, id_col="Id")

    # Query: Test Data. We need the 'Image' column (filename) for the submission file.
    # We map 'id_col' to 'Image' so extract_embeddings returns filenames in the 'ids' list.
    query_dataset = WhaleDataset(test_df, transform=transforms, id_col="Image")

    # Create Loaders
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 2. Model A Inference (EfficientNet-B5)
    # ---------------------------------------------------------
    print(f"\nProcessing Model A ({CFG.model_a_name})...")
    model_a = WhaleModel(CFG.model_a_name, num_classes=num_classes, pretrained=False)

    # Load weights
    checkpoint_a = torch.load(model_a_path, map_location=device)
    if "model" in checkpoint_a:
        model_a.load_state_dict(checkpoint_a["model"])
    else:
        model_a.load_state_dict(checkpoint_a)

    model_a.to(device)
    model_a.eval()

    # Extract Embeddings
    print("  Extracting Gallery Embeddings (A)...")
    gallery_emb_a, gallery_ids = get_embeddings(
        model_a, gallery_loader, device, tta=CFG.tta_flips
    )

    print("  Extracting Query Embeddings (A)...")
    query_emb_a, query_ids = get_embeddings(
        model_a, query_loader, device, tta=CFG.tta_flips
    )

    # Cleanup Model A to free memory for Model B
    del model_a, checkpoint_a
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 3. Model B Inference (EfficientNet-V2-M)
    # ---------------------------------------------------------
    print(f"\nProcessing Model B ({CFG.model_b_name})...")
    model_b = WhaleModel(CFG.model_b_name, num_classes=num_classes, pretrained=False)

    # Load weights
    checkpoint_b = torch.load(model_b_path, map_location=device)
    if "model" in checkpoint_b:
        model_b.load_state_dict(checkpoint_b["model"])
    else:
        model_b.load_state_dict(checkpoint_b)

    model_b.to(device)
    model_b.eval()

    # Extract Embeddings
    print("  Extracting Gallery Embeddings (B)...")
    gallery_emb_b, _ = get_embeddings(
        model_b, gallery_loader, device, tta=CFG.tta_flips
    )

    print("  Extracting Query Embeddings (B)...")
    query_emb_b, _ = get_embeddings(model_b, query_loader, device, tta=CFG.tta_flips)

    # Cleanup Model B
    del model_b, checkpoint_b
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 4. Similarity Fusion
    # ---------------------------------------------------------
    print("\nComputing Similarity Matrices...")

    # Compute Cosine Similarity (Dot Product of normalized embeddings)
    # Shape: (Num_Query, Num_Gallery)
    sim_a = np.dot(query_emb_a, gallery_emb_a.T)
    sim_b = np.dot(query_emb_b, gallery_emb_b.T)

    # Average Fusion
    sim_final = 0.5 * sim_a + 0.5 * sim_b

    # ---------------------------------------------------------
    # 5. Prediction & Open-Set Rejection
    # ---------------------------------------------------------
    print("Generating Predictions with Open-Set Rejection...")

    predictions = []
    gallery_ids_arr = np.array(gallery_ids)

    for i in range(len(query_ids)):
        scores = sim_final[i]

        # Identify nearest neighbor
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        # Sort indices by score descending
        sorted_indices = np.argsort(scores)[::-1]

        # Retrieve top unique IDs
        top_ids = []
        seen = set()

        for idx in sorted_indices:
            pred_id = gallery_ids_arr[idx]
            if pred_id not in seen:
                top_ids.append(pred_id)
                seen.add(pred_id)
            if len(top_ids) >= 5:
                break

        # Apply Threshold Logic
        # If best match is weak, predict 'new_whale' as the primary label
        if best_score < CFG.inference_threshold:
            pred_list = ["new_whale"] + top_ids[:4]
        else:
            pred_list = top_ids[:5]

        final_pred_str = " ".join(pred_list)

        predictions.append({"Image": query_ids[i], "Id": final_pred_str})

    # ---------------------------------------------------------
    # 6. Save Submission
    # ---------------------------------------------------------
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sub_df = pd.DataFrame(predictions)
    sub_df.to_csv(output_path, index=False)

    print(f"Submission saved successfully to {output_path}")
    print(f"Total Predictions: {len(sub_df)}")
    print("Sample:\n", sub_df.head())
