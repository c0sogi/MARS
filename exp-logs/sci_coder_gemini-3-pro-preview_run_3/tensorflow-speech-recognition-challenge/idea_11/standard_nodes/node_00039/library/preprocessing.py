import os
import pandas as pd
from library.config import path_cfg
from library.audio_processing import generate_cache


class CacheGenerator:
    """
    Orchestrates the offline feature extraction and caching process.
    Reads metadata, generates spectrograms via library functions, and saves
    updated metadata with cache paths.
    """

    def __init__(self, config=path_cfg):
        self.cfg = config

    def process_split(self, split_name, meta_path, load_cached_data=True):
        """
        Processes a single dataset split (train/val/test).

        Args:
            split_name (str): Name of the split (e.g., 'train').
            meta_path (str): Path to the original metadata CSV.
            load_cached_data (bool): Whether to skip processing if cache exists.

        Returns:
            str: Path to the saved cached metadata CSV.
        """
        print(f"Processing {split_name} split...")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        # Load metadata
        df = pd.read_csv(meta_path)

        # Generate cache (computes spectrograms and saves .npy files)
        # input_root is ./input, cache_dir is ./working/idea_11/cache
        df_cached = generate_cache(
            metadata_df=df,
            input_root=self.cfg.input_dir,
            cache_dir=self.cfg.cache_dir,
            load_cached_data=load_cached_data,
        )

        # Save updated metadata with 'cache_path' column to working directory
        output_filename = f"{split_name}_cached.csv"
        output_path = os.path.join(self.cfg.working_dir, output_filename)

        # Ensure working directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_cached.to_csv(output_path, index=False)
        print(f"Saved cached metadata for {split_name} to {output_path}")

        return output_path

    def run(self, load_cached_data=True):
        """
        Runs the caching process for Train, Validation, and Test sets.

        Args:
            load_cached_data (bool): If True, skips processing for existing cache files.

        Returns:
            dict: Dictionary containing paths to the updated metadata files.
        """
        # Ensure directories exist
        os.makedirs(self.cfg.working_dir, exist_ok=True)
        os.makedirs(self.cfg.cache_dir, exist_ok=True)

        outputs = {}

        # Process Train
        outputs["train"] = self.process_split(
            "train", self.cfg.train_meta, load_cached_data
        )

        # Process Validation
        outputs["val"] = self.process_split("val", self.cfg.val_meta, load_cached_data)

        # Process Test
        outputs["test"] = self.process_split(
            "test", self.cfg.test_meta, load_cached_data
        )

        return outputs


def run_preprocessing(load_cached_data=True):
    """
    Helper function to instantiate CacheGenerator and run the full pipeline.
    """
    generator = CacheGenerator()
    return generator.run(load_cached_data=load_cached_data)
