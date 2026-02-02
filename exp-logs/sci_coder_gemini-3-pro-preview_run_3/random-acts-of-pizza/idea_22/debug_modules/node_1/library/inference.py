import os
import pandas as pd
from library.config import Config
from library.utils import load_data
from library.feature_engineering import FeaturePipeline


def generate_submission(model, load_cache=True):
    """
    Orchestrates the generation of the submission file for the test set.

    This function:
    1. Runs the FeaturePipeline to obtain processed test features (Lexical, Behavioral, Semantic, etc.).
    2. Retrieves the corresponding request_ids from the raw test data.
    3. Invokes the model's internal method to generate stacked predictions and write the CSV.

    Args:
        model: An instance of the trained HexEnsemble class.
        load_cache (bool): If True, attempts to load features from the cache directory.
                           If False or cache miss, re-computes features.
    """
    # 1. Run FeaturePipeline on test data
    # We initialize the pipeline with the caching preference.
    # The pipeline.run() method handles the logic of checking/saving to ./working/idea_22/
    pipeline = FeaturePipeline(load_cached_data=load_cache)

    # The pipeline returns a dictionary containing 'train', 'val', and 'test' views.
    # We extract the 'test' view which contains the feature matrices (metadata, lexical, etc.)
    data_dict = pipeline.run()
    data_test = data_dict["test"]

    # 2. Load Test IDs
    # The feature matrices are numpy arrays/sparse matrices and do not hold the request_ids.
    # We load the raw test metadata to get the IDs corresponding to the test rows.
    df_test = load_data("test")
    test_ids = df_test[Config.ID_COL].values

    # 3. Use the model to generate predictions and save the file
    # The HexEnsemble.generate_submission method handles:
    # - Generating Level-1 probabilities from base learners
    # - Feeding them to the Level-2 Meta-Learner
    # - Formatting the dataframe and saving to Config.SUBMISSION_FILE_PATH
    model.generate_submission(data_test, test_ids)
