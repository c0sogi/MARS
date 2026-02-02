import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from library.utils import seed_everything

# Configuration Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_2"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class HerbariumDataset(Dataset):
    """
    Dataset class for Herbarium 2021 dataset.
    Handles image loading, transforming, and label mapping.
    """

    def __init__(self, df, transform=None, cat2idx=None, is_test=False):
        self.df = df
        self.transform = transform
        self.cat2idx = cat2idx
        self.is_test = is_test

        # Pre-compute paths to avoid os.path.join in the loop if possible,
        # but os.path.join is fast enough. We'll store the dataframe.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # metadata file_path is relative to input dir, e.g., "train/images/..."
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing images (though EDA showed none)
            # Create a black image of standard size
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return based on mode
        if self.is_test:
            # For test, we might need image_id for submission,
            # but usually dataloader just yields batches.
            # We'll return image_id to track predictions.
            return image, row["image_id"]
        else:
            # Map category_id to contiguous index
            cat_id = row["category_id"]
            label = self.cat2idx[cat_id]
            return image, label


def get_transforms(mode="train"):
    """
    Returns torchvision transforms for train/val/test.
    """
    if mode == "train":
        return T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        return T.Compose(
            [
                T.ToPILImage(),
                T.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to box
                T.CenterCrop(
                    IMG_SIZE
                ),  # Or just Resize directly if aspect ratio varies
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )


def get_label_map(train_df, load_cached_data=True, debug=False):
    """
    Computes and caches the mapping from category_id to class index.

    Returns:
        cat2idx (dict): Mapping from category_id to 0..N-1 index.
        idx2cat (dict): Mapping from 0..N-1 index to category_id.
        classes (np.array): Sorted array of unique category_ids.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "classes.npy")

    # If debug is on, we don't use cache to avoid polluting it with partial data
    if load_cached_data and not debug and os.path.exists(cache_path):
        classes = np.load(cache_path)
    else:
        classes = sorted(train_df["category_id"].unique())
        classes = np.array(classes)
        if not debug:
            np.save(cache_path, classes)

    cat2idx = {cat: i for i, cat in enumerate(classes)}
    idx2cat = {i: cat for i, cat in enumerate(classes)}

    return cat2idx, idx2cat, classes


def make_weights_for_balanced_classes(
    df, sampling_mode, load_cached_data=True, debug=False
):
    """
    Calculates weights for WeightedRandomSampler.

    Args:
        df (pd.DataFrame): Dataframe containing 'category_id'.
        sampling_mode (str): 'balanced' or 'sqrt'.
        load_cached_data (bool): Whether to use cached weights.
        debug (bool): If True, disables caching.

    Returns:
        weights (torch.DoubleTensor): Weights for each sample.
    """
    if sampling_mode is None:
        return None

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_name = f"train_weights_{sampling_mode}.npy"
    cache_path = os.path.join(CACHE_DIR, cache_name)

    # Try loading cache
    if load_cached_data and not debug and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            # Verify length matches (in case dataset changed)
            if len(weights_np) == len(df):
                return torch.from_numpy(weights_np).double()
        except Exception:
            pass  # Fallback to recomputing

    # Compute weights
    # 1. Count samples per class
    count_series = df["category_id"].value_counts().sort_index()
    # Ensure we cover all classes present in df
    # Map counts to category_id
    counts_dict = count_series.to_dict()

    # 2. Calculate weight per class
    # For 'balanced': target prob ~ 1/C -> weight ~ 1/N_c
    # For 'sqrt': target prob ~ sqrt(N_c)/Sum(sqrt) -> weight ~ 1/sqrt(N_c)
    # Explanation for sqrt:
    #   Prob(sample) = weight / Sum(weights)
    #   Prob(class) = N_c * Prob(sample) \propto N_c * weight
    #   We want Prob(class) \propto sqrt(N_c)
    #   So N_c * weight \propto sqrt(N_c)  =>  weight \propto 1/sqrt(N_c)

    class_weights = {}
    for cat, count in counts_dict.items():
        if sampling_mode == "balanced":
            class_weights[cat] = 1.0 / count
        elif sampling_mode == "sqrt":
            class_weights[cat] = 1.0 / np.sqrt(count)
        else:
            class_weights[cat] = 1.0

    # 3. Map to samples
    # Using map is faster than iterating
    weights_np = df["category_id"].map(class_weights).values.astype(np.float64)

    # Cache result
    if not debug:
        np.save(cache_path, weights_np)

    return torch.from_numpy(weights_np).double()


def get_dataloaders(
    batch_size=32,
    num_workers=4,
    sampling_mode=None,
    max_samples_per_class=None,
    load_cached_data=True,
    debug=False,
):
    """
        Creates training and validation dataloaders.

        Args:
            batch_size (int): Batch size.
            num_workers (int): Number of worker threads.
            sampling_mode (str): 'balanced', 'sqrt', or None.
            max_samples_per_class (int): If set, caps samples per class (Hard Undersampling).
            load_cached_data (bool): Use cached metadata/weights.
            debug (bool): If True, use a small subset of data.

        Returns:
            train_loader (DataLoader)
    >>>>>>> REPLACE
    <<<<<<< SEARCH
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

        if debug:
            # Filter to top 50 classes to ensure overlap and sufficient data density
            top_classes = train_df["category_id"].value_counts().head(50).index
            train_df = train_df[train_df["category_id"].isin(top_classes)].iloc[:2000]
            val_df = val_df[val_df["category_id"].isin(top_classes)].iloc[:500]

        # Get Label Mapping
    =======
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

        if debug:
            # Filter to top 50 classes to ensure overlap and sufficient data density
            top_classes = train_df["category_id"].value_counts().head(50).index
            train_df = train_df[train_df["category_id"].isin(top_classes)].iloc[:2000]
            val_df = val_df[val_df["category_id"].isin(top_classes)].iloc[:500]

        # Apply Hard Undersampling if requested (Cite solution_lesson_node_00009)
        if max_samples_per_class is not None and not debug:
            # Shuffle first to get random samples, then take head
            train_df = (
                train_df.sample(frac=1, random_state=42)
                .groupby("category_id")
                .head(max_samples_per_class)
                .reset_index(drop=True)
            )

        # Get Label Mapping
            val_loader (DataLoader)
            num_classes (int): Total number of classes.
    """
    seed_everything(42)

    # Load Metadata
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    if debug:
        # Filter to top 50 classes to ensure overlap and sufficient data density
        top_classes = train_df["category_id"].value_counts().head(50).index
        train_df = train_df[train_df["category_id"].isin(top_classes)].iloc[:2000]
        val_df = val_df[val_df["category_id"].isin(top_classes)].iloc[:500]

    # Get Label Mapping
    # We compute mapping based on train_df (or full train_df if not debug, but here we pass current df)
    # Note: If debug is True, we might miss classes.
    # For robustness in this function, we should ideally load the full mapping if possible,
    # but for debug runs, local mapping is fine.
    cat2idx, idx2cat, classes = get_label_map(train_df, load_cached_data, debug)
    num_classes = len(classes)

    # Datasets
    train_dataset = HerbariumDataset(
        train_df, transform=get_transforms("train"), cat2idx=cat2idx, is_test=False
    )
    val_dataset = HerbariumDataset(
        val_df, transform=get_transforms("val"), cat2idx=cat2idx, is_test=False
    )

    # Sampler
    sampler = None
    if sampling_mode in ["balanced", "sqrt"]:
        weights = make_weights_for_balanced_classes(
            train_df, sampling_mode, load_cached_data, debug
        )
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False  # Sampler provides shuffling
    else:
        shuffle = True

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, classes


def get_test_dataloader(
    batch_size=32, num_workers=4, load_cached_data=True, debug=False
):
    """
    Creates test dataloader and returns class mapping for submission.

    Returns:
        test_loader (DataLoader)
        idx2cat (dict): Mapping from prediction index to category_id.
    """
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    test_df = pd.read_csv(test_csv_path)

    if debug:
        test_df = test_df.iloc[:500]

    # We need the class mapping from the training set to decode predictions
    # We load the cached classes.npy generated during training setup
    cache_path = os.path.join(CACHE_DIR, "classes.npy")
    if os.path.exists(cache_path):
        classes = np.load(cache_path)
        idx2cat = {i: cat for i, cat in enumerate(classes)}
    else:
        # Fallback: if training hasn't run, we must load train.csv to get classes
        # This is expensive but necessary if called standalone
        train_csv_path = os.path.join(METADATA_DIR, "train.csv")
        train_df = pd.read_csv(train_csv_path)
        _, idx2cat, _ = get_label_map(train_df, load_cached_data=True, debug=False)

    test_dataset = HerbariumDataset(
        test_df, transform=get_transforms("test"), cat2idx=None, is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, idx2cat
