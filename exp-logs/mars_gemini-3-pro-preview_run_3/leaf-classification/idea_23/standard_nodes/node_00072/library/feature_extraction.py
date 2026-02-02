import os
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from library.config import Config
from library.utils import seed_everything


class LeafDataset(Dataset):
    """
    Dataset class to load leaf images.
    Returns PIL images to allow for geometric transformations (rotation)
    before tensor conversion.
    """

    def __init__(self, df, input_dir, img_size=224):
        self.df = df
        self.input_dir = input_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full path
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image and convert to RGB (standard for pre-trained models)
        try:
            image = Image.open(img_path).convert("RGB")
            # Resize immediately to reduce memory usage during rotation generation
            image = image.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error to prevent crash
            image = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))

        return image


def extract_features_from_batch(
    images, dinov2_model, convnext_model, dinov2_processor, convnext_cfg, device
):
    """
    Generates 12 rotated views for each image in the batch and extracts features.

    Args:
        images: List of PIL Images.
        dinov2_model: Loaded DINOv2 model.
        convnext_model: Loaded ConvNeXt model.
        dinov2_processor: DINOv2 Image Processor.
        convnext_cfg: ConvNeXt data config (mean/std).
        device: Torch device.

    Returns:
        numpy array of shape [Batch_Size, 12, Total_Dim]
    """
    batch_size = len(images)
    angles = Config.ROTATION_ANGLES  # [0, 30, ..., 330]
    num_rotations = len(angles)

    # 1. Generate Rotated Views
    # Flattened list: [Img1_0, Img1_30, ..., Img2_0, ...]
    all_rotated_imgs = []
    for img in images:
        for angle in angles:
            # Rotate with white fill (255) as background is white
            rot_img = F.rotate(img, angle, fill=(255, 255, 255))
            all_rotated_imgs.append(rot_img)

    # Total images to process = Batch_Size * 12
    # On A100, 32 * 12 = 384 images fit easily in memory for inference

    # 2. Extract DINOv2 Features (Global Geometry)
    # Processor handles normalization. We disable resizing as we already resized.
    dino_inputs = dinov2_processor(
        images=all_rotated_imgs,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    dino_inputs = {k: v.to(device) for k, v in dino_inputs.items()}

    with torch.no_grad():
        dino_outputs = dinov2_model(**dino_inputs)
        # Use CLS token (index 0)
        dino_feats = dino_outputs.last_hidden_state[:, 0, :]  # [B*12, 1024]

    # 3. Extract ConvNeXt Features (Local Texture)
    # Manual preprocessing for timm model
    mean = torch.tensor(convnext_cfg["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(convnext_cfg["std"]).view(1, 3, 1, 1).to(device)

    # Convert all PIL images to Tensor [B*12, 3, H, W] and scale to [0, 1]
    tensor_imgs = torch.stack([F.to_tensor(img) for img in all_rotated_imgs]).to(device)

    # Normalize
    conv_inputs = (tensor_imgs - mean) / std

    with torch.no_grad():
        conv_feats = convnext_model(conv_inputs)  # [B*12, 1536]

    # 4. Concatenate and Reshape
    # Combined: [B*12, 2560]
    combined_feats = torch.cat([dino_feats, conv_feats], dim=1)

    # Reshape to [Batch_Size, Num_Rotations, Feature_Dim]
    combined_feats = combined_feats.view(batch_size, num_rotations, -1)

    return combined_feats.cpu().numpy()


def extract_and_cache_features(load_cached_data=True):
    """
    Main function to extract features for Train (Train+Val) and Test sets.
    Handles caching to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (train_data, test_data)
        where train_data = (img_feats, tab_feats, labels, ids)
        and test_data = (img_feats, tab_feats, ids)
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define Cache Paths
    cache_files = {
        "train_img": Config.CACHE_TRAIN_IMG_FEATURES,
        "train_tab": Config.CACHE_TRAIN_TAB_FEATURES,
        "train_lbl": Config.CACHE_TRAIN_LABELS,
        "train_ids": Config.CACHE_TRAIN_IDS,
        "test_img": Config.CACHE_TEST_IMG_FEATURES,
        "test_tab": Config.CACHE_TEST_TAB_FEATURES,
        "test_ids": Config.CACHE_TEST_IDS,
    }

    # Check Cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading cached features from disk...")
        train_img = np.load(cache_files["train_img"])
        train_tab = np.load(cache_files["train_tab"])
        train_lbl = np.load(cache_files["train_lbl"], allow_pickle=True)
        train_ids = np.load(cache_files["train_ids"])

        test_img = np.load(cache_files["test_img"])
        test_tab = np.load(cache_files["test_tab"])
        test_ids = np.load(cache_files["test_ids"])

        return (train_img, train_tab, train_lbl, train_ids), (
            test_img,
            test_tab,
            test_ids,
        )

    print("Cache not found or invalid. Starting feature extraction...")

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # Model Initialization
    # ==========================================
    print(f"Initializing DINOv2 ({Config.MODEL_DINO})...")
    dinov2_processor = AutoImageProcessor.from_pretrained(Config.MODEL_DINO)
    dinov2_model = AutoModel.from_pretrained(Config.MODEL_DINO).to(device)
    dinov2_model.eval()

    print(f"Initializing ConvNeXt ({Config.MODEL_CONVNEXT})...")
    convnext_model = timm.create_model(
        Config.MODEL_CONVNEXT, pretrained=True, num_classes=0
    ).to(device)
    convnext_model.eval()
    convnext_cfg = timm.data.resolve_data_config(convnext_model.pretrained_cfg)

    # ==========================================
    # Data Loading
    # ==========================================
    print("Loading Metadata...")
    # Combine Train and Val for the full development set
    df_train_part = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_part = pd.read_csv(Config.VAL_METADATA_PATH)
    df_train = pd.concat([df_train_part, df_val_part], ignore_index=True)

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Identify Tabular Columns
    tab_cols = [
        c
        for c in df_train.columns
        if any(c.startswith(p) for p in Config.TABULAR_PREFIXES)
    ]
    print(f"Found {len(tab_cols)} tabular features.")

    # ==========================================
    # Extraction Loop
    # ==========================================
    def process_dataset(df, desc):
        dataset = LeafDataset(df, Config.INPUT_DIR, img_size=Config.IMG_SIZE)
        # collate_fn=lambda x: x returns a list of PIL images, which is what we want for our custom batch processor
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda x: x,
        )

        all_features = []

        for batch_imgs in tqdm(loader, desc=desc):
            feats = extract_features_from_batch(
                batch_imgs,
                dinov2_model,
                convnext_model,
                dinov2_processor,
                convnext_cfg,
                device,
            )
            all_features.append(feats)

        return np.concatenate(all_features, axis=0)

    # Process Train
    print(f"Processing Training Set ({len(df_train)} samples)...")
    train_img_feats = process_dataset(df_train, "Extracting Train")
    train_tab_feats = df_train[tab_cols].values.astype(np.float32)
    train_labels = df_train["species"].values
    train_ids = df_train["id"].values

    # Process Test
    print(f"Processing Test Set ({len(df_test)} samples)...")
    test_img_feats = process_dataset(df_test, "Extracting Test")
    test_tab_feats = df_test[tab_cols].values.astype(np.float32)
    test_ids = df_test["id"].values

    # ==========================================
    # Saving to Cache
    # ==========================================
    print("Saving features to cache...")
    np.save(cache_files["train_img"], train_img_feats)
    np.save(cache_files["train_tab"], train_tab_feats)
    np.save(cache_files["train_lbl"], train_labels)
    np.save(cache_files["train_ids"], train_ids)

    np.save(cache_files["test_img"], test_img_feats)
    np.save(cache_files["test_tab"], test_tab_feats)
    np.save(cache_files["test_ids"], test_ids)

    print("Feature extraction complete.")

    return (train_img_feats, train_tab_feats, train_labels, train_ids), (
        test_img_feats,
        test_tab_feats,
        test_ids,
    )
