import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, ConvNextModel
from tqdm import tqdm
from library.config import Config
from library.utils import setup_logging, seed_everything

# Initialize logger
logger = setup_logging()


class FeatureExtractor:
    """
    Handles loading of DINOv2 and ConvNeXt models and performing inference
    to extract features from image views.
    """

    def __init__(self):
        self.device = Config.DEVICE
        logger.info(f"Initializing FeatureExtractor on device: {self.device}")

        # Load DINOv2
        logger.info(f"Loading DINOv2 model: {Config.MODEL_DINO}")
        self.dino_processor = AutoImageProcessor.from_pretrained(Config.MODEL_DINO)
        self.dino_model = AutoModel.from_pretrained(Config.MODEL_DINO).to(self.device)
        self.dino_model.eval()

        # Load ConvNeXt
        logger.info(f"Loading ConvNeXt model: {Config.MODEL_CONVNEXT}")
        self.conv_processor = AutoImageProcessor.from_pretrained(Config.MODEL_CONVNEXT)
        self.conv_model = ConvNextModel.from_pretrained(Config.MODEL_CONVNEXT).to(
            self.device
        )
        self.conv_model.eval()

    def _preprocess_images(self, images, processor):
        """
        Preprocesses a list of PIL images for a specific model processor.
        Forces resize to Config.IMAGE_SIZE.
        """
        # Cite debug_lesson_17: Match Image Processor Size Keys to Model Requirements
        if "ConvNext" in processor.__class__.__name__:
            size_args = {"shortest_edge": Config.IMAGE_SIZE}
        else:
            size_args = {"height": Config.IMAGE_SIZE, "width": Config.IMAGE_SIZE}

        inputs = processor(
            images,
            return_tensors="pt",
            size=size_args,
        )
        return inputs.to(self.device)

    def extract_features(self, images):
        """
        Extracts features for a batch of images (e.g., 36 views of a single leaf).
        Returns tuple of (dino_embeddings, conv_embeddings).
        """
        with torch.no_grad():
            # DINOv2 Inference
            dino_inputs = self._preprocess_images(images, self.dino_processor)
            dino_outputs = self.dino_model(**dino_inputs)
            # Use pooler_output (CLS token for DINOv2)
            dino_emb = dino_outputs.pooler_output.cpu().numpy()

            # ConvNeXt Inference
            conv_inputs = self._preprocess_images(images, self.conv_processor)
            conv_outputs = self.conv_model(**conv_inputs)
            # Use pooler_output (Global Average Pooling for ConvNeXt)
            conv_emb = conv_outputs.pooler_output.cpu().numpy()

        return dino_emb, conv_emb


def get_rotated_views(image_path):
    """
    Loads an image and generates 36 rotated views (0 to 350 degrees).
    """
    full_path = os.path.join(Config.INPUT_DIR, image_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")

    # Load and convert to RGB (models expect 3 channels)
    img = Image.open(full_path).convert("RGB")

    views = []
    # Generate rotations: 0, 10, 20, ..., 350
    for angle in range(
        Config.ROTATION_START, Config.ROTATION_END, Config.ROTATION_STEP
    ):
        # Rotate with white background fill (255, 255, 255)
        # expand=False keeps original size, we want to preserve the frame
        rotated_img = img.rotate(
            angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255)
        )
        views.append(rotated_img)

    return views


def process_dataset(subset: str, load_cached_data: bool = True):
    """
    Main processing function.
    Iterates through the dataset defined by 'subset' ('train', 'val', 'test').
    Extracts 36-view features for DINOv2 and ConvNeXt.
    Caches results to disk.
    """
    seed_everything(Config.SEED)

    # Define cache paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_dino = os.path.join(cache_dir, f"{subset}_dino.npy")
    path_conv = os.path.join(cache_dir, f"{subset}_conv.npy")
    path_ids = os.path.join(cache_dir, f"{subset}_ids.npy")
    path_tab = os.path.join(cache_dir, f"{subset}_tab.npy")
    path_labels = os.path.join(cache_dir, f"{subset}_labels.npy")

    # Check cache
    files_exist = all(
        [os.path.exists(p) for p in [path_dino, path_conv, path_ids, path_tab]]
    )
    # Labels only exist for train/val
    if subset in ["train", "val"]:
        files_exist = files_exist and os.path.exists(path_labels)

    if load_cached_data and files_exist:
        logger.info(f"Loading cached features for subset: {subset}")
        dino_feats = np.load(path_dino)
        conv_feats = np.load(path_conv)
        ids = np.load(path_ids)
        tab_feats = np.load(path_tab)

        if subset in ["train", "val"]:
            labels = np.load(path_labels)
            return dino_feats, conv_feats, tab_feats, ids, labels
        return dino_feats, conv_feats, tab_feats, ids, None

    # If not cached, compute
    logger.info(f"Processing subset: {subset} (Cache miss or force reload)")

    # Load Metadata
    if subset == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif subset == "val":
        meta_path = Config.VAL_METADATA_PATH
    elif subset == "test":
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown subset: {subset}")

    df = pd.read_csv(meta_path)

    if Config.DEBUG:
        logger.info(f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        # Cite debug_lesson_12: Preserve Statistical Invariants in Debug Subsets
        # This constraint is strictly required only for the training set (StratifiedKFold).
        if "species" in df.columns and subset == "train":
            # Ensure at least 2 samples per class for StratifiedKFold
            df = df.groupby("species").head(2)
            df = df.sort_values("species")
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

            # Filter out any singleton classes that might result from the cut
            counts = df["species"].value_counts()
            valid_species = counts[counts >= 2].index
            df = df[df["species"].isin(valid_species)]
        else:
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Identify tabular columns
    margin_cols = [c for c in df.columns if c.startswith("margin")]
    shape_cols = [c for c in df.columns if c.startswith("shape")]
    texture_cols = [c for c in df.columns if c.startswith("texture")]
    feature_cols = margin_cols + shape_cols + texture_cols

    # Initialize Extractor
    extractor = FeatureExtractor()

    # Storage lists
    list_dino = []
    list_conv = []
    list_ids = []
    list_tab = []
    list_labels = []

    # Iterate over images
    # We process one image (36 views) at a time.
    # 36 views is a good batch size for inference on GPU.
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {subset}"):
        img_path = row["file_path"]
        image_id = row["id"]

        # 1. Get Views
        views = get_rotated_views(img_path)

        # 2. Extract Features (Batch of 36)
        dino_emb, conv_emb = extractor.extract_features(views)

        # 3. Store
        list_dino.append(dino_emb)  # Shape: (36, 1024)
        list_conv.append(conv_emb)  # Shape: (36, 1536)
        list_ids.append(image_id)
        list_tab.append(row[feature_cols].values.astype(np.float32))

        if "species" in row:
            list_labels.append(row["species"])

    # Convert to arrays
    arr_dino = np.array(list_dino, dtype=np.float32)  # (N, 36, 1024)
    arr_conv = np.array(list_conv, dtype=np.float32)  # (N, 36, 1536)
    arr_ids = np.array(list_ids, dtype=np.int64)
    arr_tab = np.array(list_tab, dtype=np.float32)

    logger.info(
        f"Extraction complete. Shapes: DINO {arr_dino.shape}, ConvNeXt {arr_conv.shape}"
    )

    # Save to cache
    np.save(path_dino, arr_dino)
    np.save(path_conv, arr_conv)
    np.save(path_ids, arr_ids)
    np.save(path_tab, arr_tab)

    if list_labels:
        arr_labels = np.array(list_labels)
        np.save(path_labels, arr_labels)
        return arr_dino, arr_conv, arr_tab, arr_ids, arr_labels

    return arr_dino, arr_conv, arr_tab, arr_ids, None
