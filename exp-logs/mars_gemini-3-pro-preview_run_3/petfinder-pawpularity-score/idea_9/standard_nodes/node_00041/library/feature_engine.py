import os
import torch
import numpy as np
import timm
from transformers import CLIPModel
from library.config import Config
from library.data_handling import get_pet_dataloader
from library.utils import seed_everything


class BackboneExtractor:
    """
    Handles the initialization of frozen backbone models and the extraction
    of features from dual-view images (Global and Zoomed).
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_names = Config.BACKBONES
        self.models = []

        print(f"Initializing BackboneExtractor on {self.device}...")

        # Initialize the 4 models specified in Config
        # 1. Swin Transformer (timm)
        self._load_timm_model(self.model_names[0], "swin")

        # 2. EfficientNetV2 (timm)
        self._load_timm_model(self.model_names[1], "effnet")

        # 3. DINOv2 (timm)
        self._load_timm_model(self.model_names[2], "dino")

        # 4. CLIP (transformers)
        self._load_clip_model(self.model_names[3], "clip")

    def _load_timm_model(self, model_name, alias):
        print(f"Loading {alias}: {model_name}...")

        # Cite debug_lesson_9: Differentiate Constructor Arguments for Heterogeneous Model Architectures
        # Pass img_size only to transformer models that need it for positional embedding interpolation.
        kwargs = {}
        if alias in ["swin", "dino"]:
            kwargs["img_size"] = Config.IMAGE_SIZE

        model = timm.create_model(model_name, pretrained=True, num_classes=0, **kwargs)
        model.eval()
        model.to(self.device)
        self.models.append({"model": model, "type": "timm", "name": alias})

    def _load_clip_model(self, model_name, alias):
        print(f"Loading {alias}: {model_name}...")
        # We use CLIPModel to get the projected image features
        model = CLIPModel.from_pretrained(model_name)
        model.eval()
        model.to(self.device)
        self.models.append({"model": model, "type": "transformers", "name": alias})

    def _get_embeddings(self, model_info, images):
        """
        Runs forward pass for a specific model.
        Args:
            model_info (dict): Dictionary containing model and type.
            images (torch.Tensor): Input images (B, 3, H, W).
        Returns:
            torch.Tensor: Embeddings (B, D).
        """
        model = model_info["model"]
        m_type = model_info["type"]

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if m_type == "timm":
                    # timm models with num_classes=0 return the pooled features
                    features = model(images)
                elif m_type == "transformers":
                    # CLIPModel returns CLIPOutput. get_image_features returns the projected embeddings.
                    features = model.get_image_features(pixel_values=images)
        return features

    def extract_features(self, dataloader):
        """
        Iterates over the dataloader, applies TTA, and extracts features from all backbones.

        Returns:
            dict: Dictionary containing numpy arrays for each stream, metadata, targets, and ids.
        """
        # Initialize storage for 8 streams (4 models * 2 views)
        # Order: Swin-G, Swin-Z, EffNet-G, EffNet-Z, DINO-G, DINO-Z, CLIP-G, CLIP-Z
        stream_buffers = [[] for _ in range(8)]

        meta_buffer = []
        target_buffer = []
        id_buffer = []

        print(f"Starting extraction on {len(dataloader)} batches...")

        for batch_idx, batch in enumerate(dataloader):
            # Move inputs to device
            global_view = batch["global_view"].to(self.device)
            zoomed_view = batch["zoomed_view"].to(self.device)

            # TTA: Create horizontally flipped versions
            global_flip = torch.flip(global_view, dims=[3])
            zoomed_flip = torch.flip(zoomed_view, dims=[3])

            # Concatenate original and flipped for batch processing
            # Shape becomes (2*B, 3, H, W)
            global_input = torch.cat([global_view, global_flip], dim=0)
            zoomed_input = torch.cat([zoomed_view, zoomed_flip], dim=0)

            batch_size = global_view.shape[0]

            # Iterate through each backbone
            for m_idx, model_info in enumerate(self.models):
                # --- Global View Stream ---
                g_feats_all = self._get_embeddings(model_info, global_input)
                # Average original and flipped embeddings
                g_feats = (g_feats_all[:batch_size] + g_feats_all[batch_size:]) / 2.0
                stream_buffers[m_idx * 2].append(g_feats.float().cpu().numpy())

                # --- Zoomed View Stream ---
                z_feats_all = self._get_embeddings(model_info, zoomed_input)
                # Average original and flipped embeddings
                z_feats = (z_feats_all[:batch_size] + z_feats_all[batch_size:]) / 2.0
                stream_buffers[m_idx * 2 + 1].append(z_feats.float().cpu().numpy())

            # Store non-image data
            meta_buffer.append(batch["metadata"].numpy())
            target_buffer.append(batch["target"].numpy())
            id_buffer.extend(batch["id"])

            # Debug break for quick testing if configured
            if Config.DEBUG and batch_idx >= 2:
                break

        # Aggregate results
        print("Aggregating features...")
        results = {}
        stream_names = [
            "swin_global",
            "swin_zoomed",
            "effnet_global",
            "effnet_zoomed",
            "dino_global",
            "dino_zoomed",
            "clip_global",
            "clip_zoomed",
        ]

        for i, name in enumerate(stream_names):
            results[name] = np.vstack(stream_buffers[i])

        results["metadata"] = np.vstack(meta_buffer)
        results["targets"] = np.concatenate(target_buffer)
        results["ids"] = np.array(id_buffer)

        return results


def extract_and_cache_features(mode, load_cached_data=True):
    """
    Extracts features for the specified dataset mode.
    Uses caching to avoid re-computation.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        dict: Dictionary containing feature arrays and metadata.
    """
    seed_everything(Config.SEED)

    # Define cache path
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{mode}_features_multiview.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"[{mode}] Loading cached features from {cache_path}...")
        try:
            # Load .npz file
            # allow_pickle=True is needed if object arrays (like string IDs) are stored
            with np.load(cache_path, allow_pickle=True) as data:
                return {key: data[key] for key in data.files}
        except Exception as e:
            print(f"[{mode}] Cache load failed ({e}). Re-computing...")

    # 2. Compute from scratch
    print(f"[{mode}] Computing features...")

    # Determine which metadata file to use
    if mode == "train":
        meta_path = Config.TRAIN_META_PATH
    elif mode == "val":
        meta_path = Config.VAL_META_PATH
    elif mode == "test":
        meta_path = Config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Initialize DataLoader (no shuffle needed for feature extraction)
    dataloader = get_pet_dataloader(
        meta_csv_path=meta_path, mode=mode, batch_size=32, shuffle=False
    )

    # Initialize Extractor and run
    extractor = BackboneExtractor()
    results = extractor.extract_features(dataloader)

    # 3. Save to cache
    print(f"[{mode}] Saving features to {cache_path}...")
    np.savez_compressed(cache_path, **results)

    return results
