import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import calculate_rmse


def extract_features(model, data_loader, device):
    """
    Extracts embeddings from the model backbone and collects metadata/targets.
    Cite {solution_lesson_node_00001}: Decouple feature extraction from learning.
    """
    model.eval()

    embeddings = []
    metadata = []
    targets = []
    ids = []

    with torch.no_grad():
        for batch_data in data_loader:
            images = batch_data["image"].to(device)
            features = batch_data["features"].to(device)
            target = batch_data["target"].to(device)

            # Extract image features using the backbone directly
            # The backbone was created with num_classes=0, so it returns the global pool
            # Cite {solution_lesson_node_00009}: Feature-Space Augmentation Averaging for Linear Probing
            img_emb = model.backbone(images)
            img_emb_flip = model.backbone(torch.flip(images, dims=[3]))
            img_emb = (img_emb + img_emb_flip) / 2.0

            embeddings.append(img_emb.cpu().numpy())
            metadata.append(features.cpu().numpy())
            targets.append(target.cpu().numpy())
            ids.extend(batch_data["id"])

    embeddings = np.concatenate(embeddings, axis=0)
    metadata = np.concatenate(metadata, axis=0)
    targets = np.concatenate(targets, axis=0)

    return embeddings, metadata, targets, ids
