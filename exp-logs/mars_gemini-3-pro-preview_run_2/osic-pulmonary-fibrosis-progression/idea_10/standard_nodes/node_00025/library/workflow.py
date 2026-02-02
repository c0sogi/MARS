import os
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data_pipeline import DataPipeline
from library.model_factory import train_laplace_solver, generate_submission


def run_workflow(load_cached_data=True, debug=False, debug_size=100):
    """
    Orchestrates the end-to-end pipeline: Data Processing -> Training -> Evaluation -> Submission.

    Args:
        load_cached_data (bool): If True, attempts to load processed features from disk.
                                 If False, re-runs feature extraction and processing.
        debug (bool): If True, runs the pipeline on a small subset of data for debugging.
        debug_size (int): Number of samples to use when debug is True.

    Returns:
        model: The trained LaplaceSolver instance.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Initialize Data Pipeline
    pipeline = DataPipeline()

    # 3. Process Datasets
    # The pipeline handles caching internally based on the split_name and load_cached_data flag.

    print("--- Processing Training Data ---")
    # Must set is_training=True to fit PCA and Scalers
    train_data = pipeline.process_dataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        split_name="train",
        load_cached_data=load_cached_data,
        is_training=True,
    )

    print("--- Processing Validation Data ---")
    # Uses pre-fitted PCA/Scalers from the pipeline instance
    val_data = pipeline.process_dataset(
        metadata_path=Config.VAL_METADATA_PATH,
        split_name="val",
        load_cached_data=load_cached_data,
        is_training=False,
    )

    # 4. Handle Debugging (Subsampling)
    if debug:
        print(f"DEBUG MODE ENABLED: Subsampling datasets to {debug_size} samples.")

        def slice_data_dict(data_dict, n):
            sliced = {}
            for k, v in data_dict.items():
                if v is not None and hasattr(v, "__len__") and len(v) > n:
                    sliced[k] = v[:n]
                else:
                    sliced[k] = v
            return sliced

        train_data = slice_data_dict(train_data, debug_size)
        val_data = slice_data_dict(val_data, debug_size)

    # 5. Train Model
    print("--- Training Laplace Solver ---")
    # train_laplace_solver handles the dual-model fitting (Quantile + ElasticNet)
    model = train_laplace_solver(train_data, val_data)

    # 6. High-Precision Evaluation
    # Explicitly print the validation metric with full precision
    if val_data["y"] is not None:
        print("--- Final Validation Evaluation ---")
        val_score = model.evaluate(val_data["X_fvc"], val_data["X_unc"], val_data["y"])
        print(f"Final Validation Laplace Log Likelihood: {val_score:.16f}")

    # 7. Generate Submission
    print("--- Generating Submission ---")

    # Process Test Data
    test_data = pipeline.process_dataset(
        metadata_path=Config.TEST_METADATA_PATH,
        split_name="test",
        load_cached_data=load_cached_data,
        is_training=False,
    )

    # Generate CSV
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    generate_submission(
        model=model,
        data_dict_test=test_data,
        test_metadata_path=Config.TEST_METADATA_PATH,
        output_path=submission_path,
    )

    return model
