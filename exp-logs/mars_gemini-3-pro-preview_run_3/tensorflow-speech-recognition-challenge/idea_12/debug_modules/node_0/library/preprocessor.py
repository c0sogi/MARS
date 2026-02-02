import os
import pandas as pd
import torch
from joblib import Parallel, delayed
from library.config import Config
from library.utils import set_seed
from library.audio_processor import AudioProcessor

# Global variable to hold the processor instance in each worker process
# This ensures we only initialize the heavy Torchaudio transforms once per worker
_worker_processor = None


def _init_worker_and_process(filepath, load_cached_data):
    """
    Helper function to initialize AudioProcessor once per worker and process the file.
    Executed in parallel workers.
    """
    global _worker_processor
    if _worker_processor is None:
        # Limit threads per worker to avoid oversubscription when running many processes
        torch.set_num_threads(1)
        _worker_processor = AudioProcessor()

    # Process the file (computes STFT and saves to cache)
    _worker_processor.process_file(filepath, load_cached_data=load_cached_data)


class Preprocessor:
    """
    Manages the offline feature extraction and caching for the dataset.
    """

    def __init__(self):
        set_seed(Config.SEED)

    def cache_dataset(self, load_cached_data=True):
        """
        Iterates through all metadata files (train, val, test) and caches the
        preprocessed features using AudioProcessor.

        Args:
            load_cached_data (bool): If True, skips files that already exist in cache.
                                     If False, forces re-computation.
        """
        print("Starting offline feature extraction...")

        # 1. Load Metadata
        # We need to process all files referenced in train, val, and test.
        dfs = []
        paths = [
            ("Train", Config.TRAIN_META),
            ("Val", Config.VAL_META),
            ("Test", Config.TEST_META),
        ]

        for name, path in paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                dfs.append(df)
                print(f"Loaded {name} metadata: {len(df)} files.")
            else:
                print(f"Warning: Metadata file {path} not found.")

        if not dfs:
            print("No metadata found. Skipping preprocessing.")
            return

        combined_df = pd.concat(dfs, ignore_index=True)

        # Get unique filepaths to avoid redundant processing
        unique_files = combined_df["filepath"].unique()
        print(f"Total unique files to process: {len(unique_files)}")

        # 2. Parallel Processing
        # Use all available CPUs for preprocessing to maximize speed
        n_jobs = os.cpu_count() or 1
        print(f"Using {n_jobs} workers for feature extraction.")

        # Execute parallel processing
        # We use 'loky' backend which is robust for multiprocessing with numpy/torch
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_init_worker_and_process)(fpath, load_cached_data)
            for fpath in unique_files
        )

        print("Feature extraction complete.")
