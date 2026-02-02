import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from library.config import Config
from library.utils import get_cpc_texts
from library.features import generate_structural_features


class PearsonDataset(Dataset):
    """
    PyTorch Dataset for the Phrase Similarity task.
    Handles data loading, CPC context mapping, structural feature generation,
    and tokenization with caching.
    """

    def __init__(
        self,
        mode: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = Config.max_length,
        short_name: str = "model",
        load_cached_data: bool = True,
    ):
        """
        Initializes the dataset.

        Args:
            mode (str): One of 'train', 'val', 'test'.
            tokenizer (PreTrainedTokenizerBase): The tokenizer instance to use.
            max_length (int): Maximum sequence length for tokenization.
            short_name (str): A short identifier for the model (e.g., 'deberta_v3').
                              Used to create unique cache files.
            load_cached_data (bool): If True, attempts to load processed data from cache.
        """
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.short_name = short_name
        self.cpc_texts = get_cpc_texts()

        # 1. Determine Data Path
        if mode == "train":
            self.data_path = Config.train_path
        elif mode == "val":
            self.data_path = Config.val_path
        elif mode == "test":
            self.data_path = Config.test_path
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'."
            )

        # 2. Load Data
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)

        # 3. Debugging: Subsample if configured
        if Config.debug:
            # Use a deterministic sample for reproducibility
            self.df = self.df.sample(n=100, random_state=Config.seed).reset_index(
                drop=True
            )
            print(f"[DEBUG] Subsampled {mode} data to {len(self.df)} rows.")

        # 4. Generate Structural Features
        # We use the library function which handles caching internally.
        # We append '_debug' to the split name if debugging to avoid cache collisions.
        split_name = f"{mode}_debug" if Config.debug else mode

        # Note: generate_structural_features expects the dataframe to compute features for.
        self.structural_features_df = generate_structural_features(
            self.df, split_name, load_cached_data=load_cached_data
        )

        # Extract the specific features defined in Config
        self.structural_features = self.structural_features_df[
            Config.structural_features
        ].values.astype(np.float32)

        # 5. Tokenization (with Caching)
        self.inputs = self._load_or_tokenize(load_cached_data)

        # 6. Process Labels
        self.labels = None
        self.raw_scores = None
        if "score" in self.df.columns:
            # Map float scores to integer classes for classification
            # 0.0 -> 0, 0.25 -> 1, 0.50 -> 2, 0.75 -> 3, 1.0 -> 4
            scores = self.df["score"].values
            self.labels = (scores * 4).round().astype(np.int64)
            self.raw_scores = scores.astype(np.float32)

    def _load_or_tokenize(self, load_cached_data: bool):
        """
        Loads tokenized data from cache or computes it.
        """
        # Define cache path
        cache_filename = f"{self.mode}_processed_{self.short_name}"
        if Config.debug:
            cache_filename += "_debug"
        cache_path = os.path.join(Config.working_dir, f"{cache_filename}.parquet")

        # Try loading
        if load_cached_data and os.path.exists(cache_path):
            print(
                f"Loading cached tokenized data for {self.short_name} ({self.mode}) from {cache_path}"
            )
            try:
                processed_df = pd.read_parquet(cache_path)
                if len(processed_df) == len(self.df):
                    # Convert list columns back to numpy arrays
                    return {
                        "input_ids": np.stack(processed_df["input_ids"].values),
                        "attention_mask": np.stack(
                            processed_df["attention_mask"].values
                        ),
                    }
                else:
                    print(
                        f"Cache length mismatch ({len(processed_df)} vs {len(self.df)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Compute
        print(f"Tokenizing data for {self.short_name} ({self.mode})...")

        # Prepare Input Texts
        # Strategy: Text A = "Context Description" + " " + "Anchor"
        #           Text B = "Target"
        # This allows the model to see the context modifying the anchor, compared against the target.

        contexts = [self.cpc_texts.get(c, "") for c in self.df["context"]]
        anchors = self.df["anchor"].astype(str).tolist()
        targets = self.df["target"].astype(str).tolist()

        # Combine Context and Anchor
        text_a_list = [f"{c} {a}" for c, a in zip(contexts, anchors)]
        text_b_list = targets

        # Tokenize
        tokenized = self.tokenizer(
            text_a_list,
            text_b_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",  # Return numpy arrays
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Save to Cache
        # Convert to list of arrays for storage in Parquet
        save_df = pd.DataFrame(
            {"input_ids": list(input_ids), "attention_mask": list(attention_mask)}
        )

        os.makedirs(Config.working_dir, exist_ok=True)
        try:
            save_df.to_parquet(cache_path)
            print(f"Saved tokenized data to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns a single sample.
        """
        item = {
            "input_ids": torch.tensor(self.inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.inputs["attention_mask"][idx], dtype=torch.long
            ),
            "structural_features": torch.tensor(
                self.structural_features[idx], dtype=torch.float32
            ),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            # We also return the raw score for metric calculation during validation
            item["scores"] = torch.tensor(self.raw_scores[idx], dtype=torch.float32)

        return item
