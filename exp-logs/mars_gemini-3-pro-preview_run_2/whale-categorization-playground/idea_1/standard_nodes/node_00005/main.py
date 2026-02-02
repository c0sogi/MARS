import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, load_metadata, compute_map5, predict_knn
from library.dataset import SiameseWhaleDataset, WhaleInferenceDataset
from library.model import EmbeddingNet, SiameseNet
from library.loss import ContrastiveLoss
from library.engine import train_model
from library.inference import get_embeddings, run_inference


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading metadata...")
    df_train = load_metadata("train")
    df_val = load_metadata("val")

    # Initialize Datasets
    # Cite solution_lesson_node_00001: Use unique cache names for train and val splits
    train_dataset = SiameseWhaleDataset(
        df_train, load_cached_data=True, cache_name="train_siamese_cache.npy"
    )

    # For validation loss monitoring, we use the Siamese dataset wrapper on val data
    val_dataset_pairs = SiameseWhaleDataset(
        df_val, load_cached_data=True, cache_name="val_siamese_cache.npy"
    )

    # Evaluation Datasets (for MAP@5 calculation)
    train_eval_dataset = WhaleInferenceDataset(
        df_train, load_cached_data=True, cache_name="train_eval_cache.npy"
    )
    val_eval_dataset = WhaleInferenceDataset(
        df_val, load_cached_data=True, cache_name="val_eval_cache.npy"
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset_pairs,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_eval_loader = DataLoader(
        val_eval_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Model Initialization & Training
    # ---------------------------------------------------------
    print("Initializing model...")
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net).to(device)

    criterion = ContrastiveLoss(margin=Config.MARGIN)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Simple scheduler to decay LR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("Starting training...")
    # train_model saves the best model to Config.MODEL_SAVE_PATH and returns the model with best weights
    model = train_model(
        model,
        train_loader,
        val_loader,
        train_eval_loader,
        val_eval_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
    )

    # ---------------------------------------------------------
    # 4. Validation Metric (MAP@5)
    # ---------------------------------------------------------
    print("\nCalculating Final Validation MAP@5...")

    # Generate embeddings for Training set (Reference) and Validation set (Query)
    # get_embeddings handles caching and returns (embeddings, ids, images)
    train_emb, train_ids, train_imgs = get_embeddings(
        model, df_train, device, "train_ref", load_cached_data=True
    )

    val_emb, val_ids, val_imgs = get_embeddings(
        model, df_val, device, "val_query", load_cached_data=True
    )

    # Predict using KNN
    val_preds = predict_knn(
        val_emb, train_emb, train_ids, threshold=Config.NEW_WHALE_THRESHOLD
    )

    # Compute Metric
    # Ensure alignment: val_ids from get_embeddings corresponds to val_imgs order
    map5_score = compute_map5(val_ids, val_preds)
    print(f"Final Validation Metric: {map5_score}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # 1. Calculate per-sample error
    # Score is 1/(rank+1) if correct in top 5, else 0. Error = 1 - Score.
    errors = []
    for target, preds in zip(val_ids, val_preds):
        score = 0.0
        if target in preds[:5]:
            rank = preds[:5].index(target)
            score = 1.0 / (rank + 1)
        errors.append(1.0 - score)

    # 2. Gather Metadata Features
    # We need to compute/lookup features for the validation images
    # Class Frequency map
    train_class_counts = df_train["Id"].value_counts().to_dict()

    analysis_data = []

    for idx, img_name in enumerate(val_imgs):
        # Find row in df_val to get file path
        # Optimization: Create a lookup dict first
        pass

    # Create lookup for val paths
    val_path_map = dict(zip(df_val["Image"], df_val["file_path"]))

    widths = []
    heights = []
    file_sizes = []
    class_freqs = []

    for i, img_name in enumerate(val_imgs):
        rel_path = val_path_map.get(img_name)
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # File Size
        if os.path.exists(full_path):
            f_size = os.path.getsize(full_path)
            # Read dims (cached in memory ideally, but fast enough to read header or use cv2)
            # We can use cv2.imread just for shape
            img = cv2.imread(full_path)
            if img is not None:
                h, w = img.shape[:2]
            else:
                h, w = 0, 0
        else:
            f_size = 0
            h, w = 0, 0

        widths.append(w)
        heights.append(h)
        file_sizes.append(f_size)

        # Class Frequency
        target_id = val_ids[i]
        freq = train_class_counts.get(target_id, 0)
        class_freqs.append(freq)

    # Create DataFrame
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "file_size": file_sizes,
            "class_freq": class_freqs,
        }
    )

    # Compute Correlations
    correlations = df_analysis.corr()["error"].drop("error")

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    if map5_score > 0.1663:
        print("\nGenerating Submission...")
        # run_inference loads the best model from disk and generates submission.csv
        run_inference(load_cached_data=True, threshold=Config.NEW_WHALE_THRESHOLD)
    else:
        print(
            f"\nSkipping submission: MAP@5 {map5_score:.4f} did not improve baseline (0.1663)"
        )


if __name__ == "__main__":
    main()
