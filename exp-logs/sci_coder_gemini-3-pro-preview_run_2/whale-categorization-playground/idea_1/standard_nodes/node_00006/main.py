import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, load_metadata, compute_map5
from library.dataset import SiameseWhaleDataset
from library.model import EmbeddingNet, SiameseNet
from library.loss import ContrastiveLoss
from library.engine import train_model
from library.inference import get_embeddings, predict_knn, run_inference


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
    # We use the full training set. Size is ~6.8k, which is small enough for fast training.
    # Cite solution_lesson_node_00001: Parameterize cache name to avoid collision
    train_dataset = SiameseWhaleDataset(
        df_train, load_cached_data=True, cache_name="train_siamese_cache.npy"
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
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

    # Define Validation Function (Cite solution_lesson_node_00002)
    def validation_fn(current_model):
        # Filter df_train for reference (exclude new_whale for reference database)
        df_train_ref = df_train[df_train["Id"] != "new_whale"]

        # Get embeddings (Cite solution_lesson_node_00003: Disable cache to get fresh embeddings)
        train_emb, train_ids, _ = get_embeddings(
            current_model, df_train_ref, device, "val_train_ref", load_cached_data=False
        )
        val_emb, val_ids, val_imgs = get_embeddings(
            current_model, df_val, device, "val_query_epoch", load_cached_data=False
        )

        preds = predict_knn(
            val_emb,
            train_emb,
            train_ids,
            val_imgs,
            threshold=Config.NEW_WHALE_THRESHOLD,
        )
        return compute_map5(val_ids, preds)

    print("Starting training...")
    # train_model saves the best model to Config.MODEL_SAVE_PATH and returns the model with best weights
    model = train_model(
        model,
        train_loader,
        None,  # val_loader not used with validation_fn
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        validation_fn=validation_fn,
    )

    # ---------------------------------------------------------
    # 4. Validation Metric (MAP@5)
    # ---------------------------------------------------------
    print("\nCalculating Validation MAP@5...")

    # Generate embeddings for Training set (Reference) and Validation set (Query)
    # Cite solution_lesson_node_00003: Disable cache to ensure we evaluate the final model
    df_train_ref = df_train[df_train["Id"] != "new_whale"]

    train_emb, train_ids, train_imgs = get_embeddings(
        model, df_train_ref, device, "train_ref_final", load_cached_data=False
    )

    val_emb, val_ids, val_imgs = get_embeddings(
        model, df_val, device, "val_query_final", load_cached_data=False
    )

    # Predict using KNN
    val_preds = predict_knn(
        val_emb, train_emb, train_ids, val_imgs, threshold=Config.NEW_WHALE_THRESHOLD
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
    if map5_score > 0.266371027346637:
        print("\nGenerating Submission...")
        # run_inference loads the best model from disk and generates submission.csv
        # Cite solution_lesson_node_00003: Disable cache
        run_inference(load_cached_data=False, threshold=Config.NEW_WHALE_THRESHOLD)
    else:
        print(
            f"\nSkipping submission: Validation score {map5_score:.4f} did not beat threshold 0.2664."
        )


if __name__ == "__main__":
    main()
