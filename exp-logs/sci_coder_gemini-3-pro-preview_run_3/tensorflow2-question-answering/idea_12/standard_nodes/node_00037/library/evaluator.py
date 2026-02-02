import os
import pandas as pd
import torch
import numpy as np
import random

from library.config import Config
from library.data_loader import load_vocab
from library.ranker_net import prepare_ranker_data, predict_test_candidates
from library.reader_net import prepare_reader_test_data, generate_submission


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_submission(load_cached_data=True):
    """
    Runs the full inference pipeline to generate the submission file.

    Args:
        load_cached_data (bool): If True, attempts to use cached intermediate files
                                 (ranker inputs, ranker outputs, reader inputs) to save time.
                                 If False, re-processes everything from scratch.
    """
    # 1. Setup and Seeding
    Config.setup_directories()
    set_seed(Config.SEED)

    print("Starting submission generation pipeline...")

    # 2. Load Vocabulary
    try:
        vocab = load_vocab()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Ranker Inference Pipeline
    # We check if the final output of the ranker (best candidates) exists.
    # If it exists and caching is enabled, we skip the ranker inference entirely.

    ranker_output_exists = os.path.exists(Config.RANKER_TEST_PATH)

    if load_cached_data and ranker_output_exists:
        print(
            f"Found cached ranker predictions at {Config.RANKER_TEST_PATH}. Skipping Ranker inference."
        )
    else:
        print("Running Ranker pipeline...")

        # Load Test Metadata
        if not os.path.exists(Config.TEST_METADATA_PATH):
            print(f"Error: Test metadata not found at {Config.TEST_METADATA_PATH}")
            return

        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

        # Prepare Ranker Inputs (Candidates)
        # We cache this intermediate step to avoid re-parsing the large JSONL file if ranker inference fails/re-runs
        ranker_inputs_cache_path = os.path.join(
            Config.WORKING_DIR, "ranker_test_inputs.parquet"
        )

        ranker_test_df = prepare_ranker_data(
            metadata_df=test_meta,
            vocab=vocab,
            raw_file_path=Config.TEST_RAW_FILE,
            is_train=False,
            load_cached_data=load_cached_data,
            cache_path=ranker_inputs_cache_path,
            perform_sampling=False,
        )

        # Run Ranker Model
        # This function saves the best candidates to Config.RANKER_TEST_PATH
        if os.path.exists(Config.RANKER_MODEL_PATH):
            predict_test_candidates(ranker_test_df, vocab)
        else:
            print(
                f"Error: Ranker model not found at {Config.RANKER_MODEL_PATH}. Cannot proceed."
            )
            return

    # 4. Reader Inference Pipeline
    print("Running Reader pipeline...")

    # Prepare Reader Inputs (Question + Top Candidate)
    # This reads from Config.RANKER_TEST_PATH
    reader_test_df = prepare_reader_test_data(
        ranker_output_path=Config.RANKER_TEST_PATH,
        vocab=vocab,
        load_cached_data=load_cached_data,
        cache_path=Config.READER_TEST_PATH,
    )

    if reader_test_df.empty:
        print("Error: Reader test data is empty. Check ranker output.")
        return

    # Run Reader Model and Generate CSV
    if os.path.exists(Config.READER_MODEL_PATH):
        generate_submission(reader_test_df, vocab)
    else:
        print(
            f"Error: Reader model not found at {Config.READER_MODEL_PATH}. Cannot generate submission."
        )
        return

    print("Pipeline completed successfully.")
