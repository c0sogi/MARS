import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import PawpularityDataset
from library.utils import save_array, load_array, check_cache_exists, set_seed


def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL images and stack other tensors.
    """
    ids = [item["id"] for item in batch]
    images = [item["image"] for item in batch]  # Keep as list of PIL Images
    metas = torch.stack([item["meta"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])

    return {"id": ids, "image": images, "meta": metas, "target": targets}


def get_embedding(model, inputs, backbone_name):
    """
    Extracts embeddings based on the specific architecture of the backbone.
    """
    if "siglip" in backbone_name:
        # SigLIP: Use get_image_features for projected semantic embeddings
        # SiglipModel usually provides this method
        if hasattr(model, "get_image_features"):
            return model.get_image_features(**inputs)
        else:
            # Fallback if loaded as vision model only
            outputs = model(**inputs)
            # Use pooler_output if available, else CLS or mean pooling
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                return outputs.last_hidden_state[:, 0, :]  # CLS token assumption

    elif "dinov2" in backbone_name:
        # DINOv2: Use CLS token from last_hidden_state
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    elif "convnext" in backbone_name:
        # ConvNeXt: Use pooler_output (global average pooling)
        outputs = model(**inputs)
        return outputs.pooler_output

    else:
        # Generic fallback: Pooler output or CLS
        outputs = model(**inputs)
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]


def extract_features_for_split(
    backbone_name,
    backbone_key,
    split_name,
    csv_path,
    model,
    processor,
    device,
    debug,
    load_cached_data,
):
    """
    Extracts features for a specific dataset split (train/val/test) using a specific backbone.
    Handles caching logic.
    """
    # Define cache filenames
    prefix = f"{backbone_key}_{split_name}"
    files = {
        "features": f"{prefix}_features.npy",
        "ids": f"{prefix}_ids.npy",
        "meta": f"{prefix}_meta.npy",
        "targets": f"{prefix}_targets.npy",
    }

    # Check cache
    all_cached = all(check_cache_exists(f) for f in files.values())

    if load_cached_data and all_cached:
        print(f"Loading cached features for {backbone_key} - {split_name}...")
        return {k: load_array(v) for k, v in files.items()}

    print(f"Extracting features for {backbone_key} - {split_name}...")

    # Initialize Dataset
    # We pass transform=None to get raw PIL images for flip augmentation
    dataset = PawpularityDataset(csv_path=csv_path, transform=None, debug=debug)

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    all_features = []
    all_ids = []
    all_meta = []
    all_targets = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(
            dataloader, disable=True
        ):  # Suppress progress bar as requested
            ids = batch["id"]
            images = batch["image"]  # List of PIL images
            meta = batch["meta"]
            targets = batch["target"]

            # 1. Process Original Images
            inputs_orig = processor(images=images, return_tensors="pt").to(device)

            # 2. Process Flipped Images (Feature-Space Augmentation)
            if Config.USE_FLIP_AUGMENTATION:
                images_flipped = [
                    img.transpose(Image.FLIP_LEFT_RIGHT) for img in images
                ]
                inputs_flip = processor(images=images_flipped, return_tensors="pt").to(
                    device
                )

            # 3. Extract Embeddings
            emb_orig = get_embedding(model, inputs_orig, backbone_name)

            if Config.USE_FLIP_AUGMENTATION:
                emb_flip = get_embedding(model, inputs_flip, backbone_name)
                # Average embeddings
                emb_final = (emb_orig + emb_flip) / 2.0
            else:
                emb_final = emb_orig

            # Move to CPU and store
            all_features.append(emb_final.cpu().numpy())
            all_ids.extend(ids)
            all_meta.append(meta.numpy())
            all_targets.append(targets.numpy())

    # Concatenate results
    results = {
        "features": np.concatenate(all_features, axis=0),
        "ids": np.array(all_ids),
        "meta": np.concatenate(all_meta, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
    }

    # Save to cache
    print(f"Saving features for {backbone_key} - {split_name} to {Config.CACHE_DIR}...")
    save_array(files["features"], results["features"])
    save_array(files["ids"], results["ids"])
    save_array(files["meta"], results["meta"])
    save_array(files["targets"], results["targets"])

    return results


def run_feature_extraction(debug: bool = False, load_cached_data: bool = True):
    """
    Main driver function to extract features for all backbones and all splits.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    splits = [
        ("train", Config.TRAIN_METADATA),
        ("val", Config.VAL_METADATA),
        ("test", Config.TEST_METADATA),
    ]

    # Iterate over backbones defined in Config
    for friendly_name, model_id in Config.BACKBONES.items():
        print(f"\n=== Processing Backbone: {friendly_name} ({model_id}) ===")

        # Check if all splits are already cached for this backbone to avoid loading model unnecessarily
        # This is an optimization; if load_cached_data is False, we must process anyway.
        all_splits_cached = True
        if load_cached_data:
            for split_name, _ in splits:
                prefix = f"{friendly_name}_{split_name}"
                if not check_cache_exists(f"{prefix}_features.npy"):
                    all_splits_cached = False
                    break

        if load_cached_data and all_splits_cached:
            print(f"All files cached for {friendly_name}. Skipping model loading.")
            continue

        # Load Model and Processor
        print(f"Loading model: {model_id}")
        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            raise e

        # Process each split
        for split_name, csv_path in splits:
            extract_features_for_split(
                backbone_name=model_id,
                backbone_key=friendly_name,
                split_name=split_name,
                csv_path=csv_path,
                model=model,
                processor=processor,
                device=device,
                debug=debug,
                load_cached_data=load_cached_data,
            )

        # Cleanup to free GPU memory for next backbone
        del model
        del processor
        torch.cuda.empty_cache()
        print(f"Finished processing {friendly_name}. Memory cleared.")

    print("\nFeature extraction complete.")
