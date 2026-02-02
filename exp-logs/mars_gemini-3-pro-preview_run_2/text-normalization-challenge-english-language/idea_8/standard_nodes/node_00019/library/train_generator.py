import torch
from library.config import cfg
from library.data_utils import load_generator_data, TextNormalizationDataset
from library.modeling import GeneratorModel


def _slice_dataset(dataset, num_samples):
    """
    Helper function to slice a TextNormalizationDataset for debugging purposes.

    Args:
        dataset (TextNormalizationDataset): The original dataset.
        num_samples (int): The number of samples to keep.

    Returns:
        TextNormalizationDataset: A new dataset instance with sliced data.
    """
    # Slice encodings (dict of tensors/lists)
    new_encodings = {k: v[:num_samples] for k, v in dataset.encodings.items()}

    # Slice labels if they exist
    new_labels = None
    if dataset.labels is not None:
        new_labels = dataset.labels[:num_samples]

    return TextNormalizationDataset(new_encodings, new_labels)


def run_generator_training(debug=False, load_cached_data=True):
    """
    Main function to train the Neural Generator (Seq2Seq) model.
    Focuses on complex semiotic classes (DATE, MEASURE, etc.) using context-augmented inputs.

    Args:
        debug (bool): If True, runs with a small subset of data and 1 epoch for testing.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
    """
    print(f"Starting Generator Training (Debug={debug})...")

    # 1. Load Data
    # The load_generator_data function handles caching internally based on cfg.CACHE_DIR
    # It filters the dataset to only include NEURAL_BASED_CLASSES and constructs context windows.
    train_dataset = load_generator_data(
        split="train", load_cached_data=load_cached_data
    )
    val_dataset = load_generator_data(split="val", load_cached_data=load_cached_data)

    # 2. Handle Debug Mode
    if debug:
        print("Debug mode enabled: Reducing dataset size and epochs.")
        debug_size = 1000
        # Override config for this run
        cfg.GENERATOR_EPOCHS = 1

        train_dataset = _slice_dataset(train_dataset, debug_size)
        val_dataset = _slice_dataset(val_dataset, debug_size)

        print(f"Train samples (sliced): {len(train_dataset)}")
        print(f"Val samples (sliced):   {len(val_dataset)}")
    else:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples:   {len(val_dataset)}")

    # 3. Initialize Model
    # This loads the ByT5-small backbone
    model = GeneratorModel()

    # 4. Execute Training
    # The train method handles the loop, optimizer, scheduler, validation, and early stopping
    model.train(train_dataset, val_dataset)

    print("Generator training completed.")
