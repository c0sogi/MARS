import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.model import WhaleEmbeddingNet, ContrastiveLoss
from library.dataset import WhalePairsDataset, WhaleInferenceDataset, get_transforms
from library.utils import seed_everything


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for img1, img2, target in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output1 = model(img1)
        output2 = model(img2)

        loss = criterion(output1, output2, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img1.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set using Contrastive Loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        for img1, img2, target in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            target = target.to(device)

            output1 = model(img1)
            output2 = model(img2)

            loss = criterion(output1, output2, target)

            running_loss += loss.item() * img1.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def train_model(
    train_csv=Config.TRAIN_CSV,
    val_csv=Config.VAL_CSV,
    output_path=Config.MODEL_PATH,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    device=Config.DEVICE,
    subset_size=Config.DEBUG_SUBSET_SIZE,
    patience=Config.PATIENCE,
):
    """
    Main training loop with Early Stopping.
    """
    seed_everything(Config.SEED)

    # 1. Prepare DataLoaders
    train_dataset = WhalePairsDataset(
        csv_file=train_csv,
        subset_size=subset_size,
        transform=get_transforms(mode="train"),
    )
    val_dataset = WhalePairsDataset(
        csv_file=val_csv, subset_size=subset_size, transform=get_transforms(mode="val")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Setup Model and Optimizer
    model = WhaleEmbeddingNet(embedding_dim=Config.EMBEDDING_DIM)
    model.to(device)

    criterion = ContrastiveLoss(margin=Config.MARGIN)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training complete. Best Val Loss: {best_val_loss}")
    return model


def generate_embeddings(model, dataloader, device):
    """
    Helper function to pass a dataset through the model and retrieve embeddings.
    Returns: embeddings (np.array), image_names (list), ids (list)
    """
    model.eval()
    all_embeddings = []
    all_names = []
    all_ids = []

    with torch.no_grad():
        for imgs, names, ids in dataloader:
            imgs = imgs.to(device)
            embeddings = model.get_embedding(imgs)

            all_embeddings.append(embeddings.cpu().numpy())
            all_names.extend(names)
            all_ids.extend(ids)

    if len(all_embeddings) > 0:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        all_embeddings = np.array([])

    return all_embeddings, all_names, all_ids


def predict_submission(
    model_path=Config.MODEL_PATH,
    train_csv=Config.TRAIN_CSV,
    test_csv=Config.TEST_CSV,
    submission_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    subset_size=Config.DEBUG_SUBSET_SIZE,
    load_cached_data=False,
):
    """
    Generates the submission file using the trained model.
    Constructs a gallery from the training set and queries it with the test set.
    """
    seed_everything(Config.SEED)

    # Setup Model
    model = WhaleEmbeddingNet(embedding_dim=Config.EMBEDDING_DIM)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Ensure cache directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Gallery Generation (Train Data)
    # ---------------------------------------------------------
    gallery_emb_path = os.path.join(cache_dir, "gallery_embeddings.npy")
    gallery_ids_path = os.path.join(cache_dir, "gallery_ids.npy")

    if (
        load_cached_data
        and os.path.exists(gallery_emb_path)
        and os.path.exists(gallery_ids_path)
    ):
        print("Loading cached gallery embeddings...")
        gallery_embeddings = np.load(gallery_emb_path)
        gallery_ids = np.load(gallery_ids_path)
    else:
        print("Generating gallery embeddings...")
        # Prepare Gallery Data: Exclude 'new_whale'
        df_train = pd.read_csv(train_csv)
        df_gallery = df_train[df_train["Id"] != "new_whale"].reset_index(drop=True)

        # Save temporary CSV for the Dataset class
        temp_gallery_csv = os.path.join(cache_dir, "temp_gallery.csv")
        df_gallery.to_csv(temp_gallery_csv, index=False)

        gallery_dataset = WhaleInferenceDataset(
            csv_file=temp_gallery_csv,
            subset_size=subset_size,
            transform=get_transforms(mode="test"),
        )
        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        gallery_embeddings, _, gallery_ids = generate_embeddings(
            model, gallery_loader, device
        )
        gallery_ids = np.array(gallery_ids)

        # Save to cache
        np.save(gallery_emb_path, gallery_embeddings)
        np.save(gallery_ids_path, gallery_ids)

        # Clean up temp file
        if os.path.exists(temp_gallery_csv):
            os.remove(temp_gallery_csv)

    # ---------------------------------------------------------
    # 2. Query Generation (Test Data)
    # ---------------------------------------------------------
    test_emb_path = os.path.join(cache_dir, "test_embeddings.npy")
    test_names_path = os.path.join(cache_dir, "test_names.npy")

    if (
        load_cached_data
        and os.path.exists(test_emb_path)
        and os.path.exists(test_names_path)
    ):
        print("Loading cached test embeddings...")
        test_embeddings = np.load(test_emb_path)
        test_names = np.load(test_names_path)
    else:
        print("Generating test embeddings...")
        test_dataset = WhaleInferenceDataset(
            csv_file=test_csv,
            subset_size=subset_size,
            transform=get_transforms(mode="test"),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_embeddings, test_names, _ = generate_embeddings(model, test_loader, device)
        test_names = np.array(test_names)

        # Save to cache
        np.save(test_emb_path, test_embeddings)
        np.save(test_names_path, test_names)

    # ---------------------------------------------------------
    # 3. Distance Calculation & Prediction
    # ---------------------------------------------------------
    print("Computing distances and generating predictions...")

    # Move to GPU for distance calculation
    q_tensor = torch.from_numpy(test_embeddings).to(device)
    g_tensor = torch.from_numpy(gallery_embeddings).to(device)

    # Compute pairwise Euclidean distances
    # Shape: (Num_Test_Images, Num_Gallery_Images)
    dists = torch.cdist(q_tensor, g_tensor, p=2)

    # Find Top K nearest neighbors
    k = min(Config.KNN_K, len(gallery_ids))
    topk_vals, topk_indices = torch.topk(dists, k=k, dim=1, largest=False)

    # Move back to CPU
    topk_vals = topk_vals.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()

    submission_rows = []

    for i in range(len(test_names)):
        img_name = test_names[i]

        neighbor_indices = topk_indices[i]
        neighbor_dists = topk_vals[i]

        # Map indices to IDs
        neighbor_ids = gallery_ids[neighbor_indices]

        # Collect unique IDs (preserving order)
        unique_ids = []
        seen = set()
        for nid in neighbor_ids:
            if nid not in seen:
                unique_ids.append(nid)
                seen.add(nid)
            if len(unique_ids) >= 5:
                break

        # Threshold Logic
        nearest_dist = neighbor_dists[0]
        final_preds = []

        if nearest_dist > Config.NEW_WHALE_THRESHOLD:
            # Primary prediction is new_whale
            final_preds.append("new_whale")
            final_preds.extend(unique_ids)
        else:
            # Primary prediction is the nearest neighbor
            final_preds.extend(unique_ids)
            final_preds.append("new_whale")

        # Ensure exactly 5 unique predictions
        clean_preds = []
        seen_final = set()
        for p in final_preds:
            if p not in seen_final:
                clean_preds.append(p)
                seen_final.add(p)
            if len(clean_preds) == 5:
                break

        pred_str = " ".join(clean_preds)
        submission_rows.append({"Image": img_name, "Id": pred_str})

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
