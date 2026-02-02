import pandas as pd
from library.config import Config


class TabularDataset:
    """
    Simple loader for tabular regression features used by LightGBM.
    """

    def __init__(self, split: str):
        if split == "train":
            self.path = Config.TRAIN_TABULAR_PATH
        elif split == "val":
            self.path = Config.VAL_TABULAR_PATH
        elif split == "test":
            self.path = Config.TEST_TABULAR_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

    def load(self):
        print(f"Loading tabular data from {self.path}...")
        df = pd.read_parquet(self.path)

        feature_cols = [
            "n_code",
            "sim_max",
            "sim_mean",
            "sim_std",
            "best_match_loc",
            "center_of_mass",
        ]

        X = df[feature_cols]
        y = df["target"]
        meta = df[["notebook_id", "cell_id"]]

        return X, y, meta
