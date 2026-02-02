import os
import numpy as np
import pandas as pd
from library.utils import to_float64
from library.image_processor import extract_morphometrics


class DataManager:
    def __init__(self, cache_dir="./working/idea_57"):
        """
        Initializes the DataManager with a cache directory.

        Args:
            cache_dir (str): Directory to store processed feature arrays.
        """
        self.cache_dir = cache_dir
        self.metadata_dir = "./metadata"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_data(self, load_cached_data=True):
        """
        Main method to load and process all data splits (train, val, test).

        Args:
            load_cached_data (bool): If True, attempts to load from .npz cache.

        Returns:
            dict: A dictionary containing 'train', 'val', 'test' data dictionaries and 'classes'.
        """
        # Define cache paths
        cache_files = {
            "train": os.path.join(self.cache_dir, "data_train.npz"),
            "val": os.path.join(self.cache_dir, "data_val.npz"),
            "test": os.path.join(self.cache_dir, "data_test.npz"),
            "meta": os.path.join(self.cache_dir, "data_meta.npz"),
        }

        # 1. Try Loading from Cache
        if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
            print("Loading processed data from cache...")
            return self._load_from_cache(cache_files)

        # 2. Process from Scratch
        print("Generating data views from scratch...")

        # Load Metadata CSVs
        df_train = pd.read_csv(os.path.join(self.metadata_dir, "train.csv"))
        df_val = pd.read_csv(os.path.join(self.metadata_dir, "val.csv"))
        df_test = pd.read_csv(os.path.join(self.metadata_dir, "test.csv"))

        # Extract Morphometrics (Image Features)
        # The image processor handles its own caching for the raw extraction
        morpho_train = extract_morphometrics(
            df_train["image_path"].tolist(), "train", load_cached_data
        )
        morpho_val = extract_morphometrics(
            df_val["image_path"].tolist(), "val", load_cached_data
        )
        morpho_test = extract_morphometrics(
            df_test["image_path"].tolist(), "test", load_cached_data
        )

        # Encode Labels
        # Stratified split ensures all classes are in train
        classes = sorted(df_train["species"].unique())
        class_map = {c: i for i, c in enumerate(classes)}

        y_train = df_train["species"].map(class_map).values.astype(int)
        y_val = df_val["species"].map(class_map).values.astype(int)

        # Construct Data Dictionary
        data = {
            "classes": np.array(classes),
            "train": self._process_split(df_train, morpho_train, y_train),
            "val": self._process_split(df_val, morpho_val, y_val),
            "test": self._process_split(df_test, morpho_test, None),
        }

        # Save to Cache
        self._save_to_cache(data, cache_files)

        return data

    def get_train_val_split(self, load_cached_data=True):
        """
        Convenience method to retrieve specifically the training and validation splits.
        """
        data = self.get_data(load_cached_data=load_cached_data)
        return data["train"], data["val"], data["classes"]

    def _process_split(self, df, morpho, y=None):
        """
        Constructs the dictionary of feature views for a single dataset split.
        """

        # Helper to extract and sort columns by index
        def get_cols(prefix):
            cols = [c for c in df.columns if c.startswith(prefix)]
            # Sort by the integer suffix (e.g., margin_1, margin_2)
            cols = sorted(cols, key=lambda x: int(x.replace(prefix, "").strip("_")))
            return to_float64(df[cols].values)

        # Base Features
        margin = get_cols("margin")
        shape = get_cols("shape")
        texture = get_cols("texture")
        morpho = to_float64(morpho)

        # Construct HCIPE Views
        views = {
            "global": np.hstack([margin, shape, texture]),
            "margin": margin,
            "shape": shape,
            "texture": texture,
            "morpho": morpho,
            # Pairwise Interactions
            "margin_shape": np.hstack([margin, shape]),
            "margin_texture": np.hstack([margin, texture]),
            "shape_texture": np.hstack([shape, texture]),
        }

        split_data = {"ids": df["id"].values, "views": views}

        if y is not None:
            split_data["y"] = y

        return split_data

    def _save_to_cache(self, data, cache_files):
        """
        Saves the data dictionary to .npz files using numpy.
        """
        # Save Meta
        np.savez(cache_files["meta"], classes=data["classes"])

        # Save Splits
        for split in ["train", "val", "test"]:
            split_data = data[split]
            save_dict = {"ids": split_data["ids"]}
            if "y" in split_data:
                save_dict["y"] = split_data["y"]

            # Flatten views for saving
            for view_name, view_arr in split_data["views"].items():
                save_dict[f"view_{view_name}"] = view_arr

            np.savez(cache_files[split], **save_dict)

        print(f"Data cached in {self.cache_dir}")

    def _load_from_cache(self, cache_files):
        """
        Loads data from .npz files and reconstructs the dictionary structure.
        """
        data = {}

        # Load Meta (allow_pickle=True required for string arrays)
        with np.load(cache_files["meta"], allow_pickle=True) as meta:
            data["classes"] = meta["classes"]

        # Load Splits
        for split in ["train", "val", "test"]:
            with np.load(cache_files[split], allow_pickle=True) as loaded:
                split_dict = {"ids": loaded["ids"], "views": {}}
                if "y" in loaded:
                    split_dict["y"] = loaded["y"]

                # Reconstruct views dictionary
                for key in loaded.files:
                    if key.startswith("view_"):
                        view_name = key.replace("view_", "")
                        split_dict["views"][view_name] = to_float64(loaded[key])

                data[split] = split_dict

        return data
