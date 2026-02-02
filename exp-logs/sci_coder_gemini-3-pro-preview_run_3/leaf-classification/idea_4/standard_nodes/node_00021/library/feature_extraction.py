import os
import numpy as np
import torch
import torch.nn as nn
import timm
from library import config, utils, data_loader


class DualStreamExtractor(nn.Module):
    """
    Dual-stream feature extractor using DINOv2 (ViT) and ConvNeXt.
    Handles multi-view input by averaging embeddings across views.
    """

    def __init__(self):
        super().__init__()
        # Initialize DINOv2 (ViT-Large)
        # num_classes=0 returns the pooled features (CLS token or GAP)
        self.vit = timm.create_model(
            config.MODEL_1_NAME,
            pretrained=True,
            num_classes=0,
            img_size=config.IMG_SIZE,
        )

        # Initialize ConvNeXt Large
        self.cnn = timm.create_model(
            config.MODEL_2_NAME, pretrained=True, num_classes=0
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, 4, 3, H, W)
        Returns:
            tuple: (vit_embeddings, cnn_embeddings)
                vit_embeddings: (Batch, Vit_Dim)
                cnn_embeddings: (Batch, Cnn_Dim)
        """
        b, v, c, h, w = x.shape

        # Flatten batch and views: (B*4, 3, H, W)
        x_flat = x.view(b * v, c, h, w)

        # Extract features
        # DINOv2
        vit_out = self.vit(x_flat)  # (B*4, 1024)

        # ConvNeXt
        cnn_out = self.cnn(x_flat)  # (B*4, 1536)

        # Reshape back to (Batch, Views, Dim)
        vit_out = vit_out.view(b, v, -1)
        cnn_out = cnn_out.view(b, v, -1)

        # Average across views
        vit_avg = vit_out.mean(dim=1)
        cnn_avg = cnn_out.mean(dim=1)

        return vit_avg, cnn_avg


def extract_features(split, load_cached_data=True, batch_size=config.BATCH_SIZE):
    """
    Extracts features for a given dataset split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache if available.
        batch_size (int): Batch size for the dataloader.

    Returns:
        tuple: (cnn_emb, vit_emb, tabular, labels, ids)
            cnn_emb (np.ndarray): ConvNeXt embeddings.
            vit_emb (np.ndarray): DINOv2 embeddings.
            tabular (np.ndarray): Raw tabular features.
            labels (np.ndarray): Class labels.
            ids (np.ndarray): Image IDs.
    """
    utils.seed_everything(config.SEED)

    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    paths = {
        "cnn": os.path.join(cache_dir, f"{split}_cnn_embeddings.npy"),
        "vit": os.path.join(cache_dir, f"{split}_vit_embeddings.npy"),
        "tab": os.path.join(cache_dir, f"{split}_tabular_X.npy"),
        "lbl": os.path.join(cache_dir, f"{split}_tabular_y.npy"),
        "ids": os.path.join(cache_dir, f"{split}_ids.npy"),
    }

    # Check cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in paths.values())
        if all_exist:
            cnn_emb = np.load(paths["cnn"])
            vit_emb = np.load(paths["vit"])
            tab_feat = np.load(paths["tab"])
            labels = np.load(paths["lbl"])
            ids = np.load(paths["ids"])
            return cnn_emb, vit_emb, tab_feat, labels, ids

    # If not cached, compute
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = DualStreamExtractor()
    model.to(device)
    model.eval()

    # Get dataloaders
    # We load all but only use the one corresponding to 'split'
    train_loader, val_loader, test_loader, _ = data_loader.get_dataloaders(
        batch_size=batch_size,
        load_cached_data=True,  # Use cached tabular data inside data_loader
    )

    if split == "train":
        loader = train_loader
    elif split == "val":
        loader = val_loader
    elif split == "test":
        loader = test_loader
    else:
        raise ValueError(f"Unknown split: {split}")

    # Storage
    cnn_list = []
    vit_list = []
    tab_list = []
    lbl_list = []
    id_list = []

    with torch.no_grad():
        for images, tabular, labels, img_ids in loader:
            images = images.to(device)

            # Forward pass
            vit_out, cnn_out = model(images)

            # Move to CPU and collect
            cnn_list.append(cnn_out.cpu().numpy())
            vit_list.append(vit_out.cpu().numpy())
            tab_list.append(tabular.numpy())
            lbl_list.append(labels.numpy())
            id_list.append(img_ids.numpy())

    # Concatenate
    cnn_emb = np.concatenate(cnn_list, axis=0)
    vit_emb = np.concatenate(vit_list, axis=0)
    tab_feat = np.concatenate(tab_list, axis=0)
    labels = np.concatenate(lbl_list, axis=0)
    ids = np.concatenate(id_list, axis=0)

    # Save to cache
    np.save(paths["cnn"], cnn_emb)
    np.save(paths["vit"], vit_emb)
    np.save(paths["tab"], tab_feat)
    np.save(paths["lbl"], labels)
    np.save(paths["ids"], ids)

    return cnn_emb, vit_emb, tab_feat, labels, ids
