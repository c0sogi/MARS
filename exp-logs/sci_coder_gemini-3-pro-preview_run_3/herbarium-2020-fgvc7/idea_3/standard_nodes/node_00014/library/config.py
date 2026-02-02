import os
import json
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
TRAIN_METADATA_JSON = os.path.join(INPUT_DIR, "nybg2020/train/metadata.json")

# Output Paths
MAPPING_CACHE_PATH = os.path.join(WORKING_DIR, "category_mappings.parquet")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "resnet50_arcface_mtl_best.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
BACKBONE = "resnet50"
EMBEDDING_DIM = 512
NUM_CLASSES = 32093  # Number of species (target)
# NUM_GENUS_CLASSES is determined dynamically via get_mappings()

# ArcFace Parameters
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.50

# Multi-Task Learning Parameters
LAMBDA_GENUS = 0.5  # Weight for auxiliary genus classification loss

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64  # Adjusted for A100 (40GB) + ResNet50
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda"

# -----------------------------------------------------------------------------
# Data Preprocessing
# -----------------------------------------------------------------------------
IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Data Processing & Caching Logic
# -----------------------------------------------------------------------------
def get_mappings(load_cached=True):
    """
    Generates or loads mappings for Species (ID -> Index) and Genus (ID -> Index).

    Returns:
        mapping_df (pd.DataFrame): DataFrame containing:
            - category_id: Original species ID
            - species_idx: Mapped 0..N-1 species index
            - genus: Genus name
            - genus_idx: Mapped 0..M-1 genus index
        num_species (int): Total number of species
        num_genus (int): Total number of genera
    """
    # 1. Try to load cached data
    if load_cached and os.path.exists(MAPPING_CACHE_PATH):
        print(f"Loading cached mappings from {MAPPING_CACHE_PATH}...")
        mapping_df = pd.read_parquet(MAPPING_CACHE_PATH)
        num_species = mapping_df["species_idx"].max() + 1
        num_genus = mapping_df["genus_idx"].max() + 1
        return mapping_df, num_species, num_genus

    print("Computing mappings from scratch...")

    # 2. Load Species List from Train CSV (contains all classes including singletons)
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Train CSV not found at {TRAIN_CSV}")

    train_df = pd.read_csv(TRAIN_CSV)
    unique_species = sorted(train_df["category_id"].unique())

    # Create Species Mapping (category_id -> species_idx)
    species_map_df = pd.DataFrame(
        {"category_id": unique_species, "species_idx": np.arange(len(unique_species))}
    )

    # 3. Load Genus Information from Raw Metadata JSON
    print(f"Parsing categories from {TRAIN_METADATA_JSON}...")
    with open(TRAIN_METADATA_JSON, "r") as f:
        # We only need the 'categories' list.
        data = json.load(f)
        categories_list = data["categories"]

    categories_df = pd.DataFrame(categories_list)
    # categories_df columns: ['id', 'name', 'family', 'genus']
    # Rename 'id' to 'category_id' for merging
    categories_df = categories_df.rename(columns={"id": "category_id"})

    # 4. Merge Species Map with Genus Info
    # We only care about species present in our training set
    full_mapping = pd.merge(
        species_map_df,
        categories_df[["category_id", "genus"]],
        on="category_id",
        how="left",
    )

    # Handle missing genus if any
    if full_mapping["genus"].isnull().any():
        print(
            "Warning: Some species missing genus information. Filling with 'unknown'."
        )
        full_mapping["genus"] = full_mapping["genus"].fillna("unknown")

    # 5. Create Genus Mapping (genus_string -> genus_idx)
    unique_genera = sorted(full_mapping["genus"].unique())
    genus_map = {name: idx for idx, name in enumerate(unique_genera)}

    full_mapping["genus_idx"] = full_mapping["genus"].map(genus_map)

    # 6. Save to Cache
    print(f"Saving mappings to {MAPPING_CACHE_PATH}...")
    full_mapping.to_parquet(MAPPING_CACHE_PATH, index=False)

    num_species = len(unique_species)
    num_genus = len(unique_genera)

    print(f"Mapping Complete: {num_species} Species, {num_genus} Genera.")

    return full_mapping, num_species, num_genus
