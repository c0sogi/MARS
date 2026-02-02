import os
from library.config import Config
from library.engine import predict_submission


def predict(models_config=None, debug=False):
    """
    Orchestrates the inference process for the test set using the trained ensemble.

    This function identifies available model checkpoints, configures the environment,
    and delegates the prediction generation to the engine.

    Args:
        models_config (list, optional): A list of tuples (model_name, checkpoint_path).
                                        If None, the function scans the working directory
                                        defined in Config for checkpoints matching the
                                        ensemble configuration.
        debug (bool): If True, enables debug mode which processes a small subset
                      of the test data for verification purposes.
    """
    # 1. Configure Debug Mode
    if debug:
        Config.DEBUG = True
        print("Debug mode enabled: Inference will run on a subset of the test data.")

    # 2. Resolve Model Configuration
    if models_config is None:
        print(f"Scanning {Config.WORK_DIR} for trained model checkpoints...")
        models_config = []

        # Iterate through all defined architectures and folds to find trained weights
        for model_name in Config.MODEL_ARCHS:
            for fold_idx in range(Config.N_FOLDS):
                # Construct expected checkpoint path based on engine.py naming convention
                checkpoint_name = f"{model_name}_fold{fold_idx}_best.pth"
                checkpoint_path = os.path.join(Config.WORK_DIR, checkpoint_name)

                if os.path.exists(checkpoint_path):
                    models_config.append((model_name, checkpoint_path))
                    print(f"  Found: {checkpoint_name}")
                else:
                    # Log missing models but continue with available ones
                    print(f"  Missing: {checkpoint_name} (Skipping)")

    # 3. Validation
    if not models_config:
        print("Error: No trained models found. Cannot generate submission.")
        return

    print(f"Starting inference with ensemble of {len(models_config)} models.")

    # 4. Execute Inference Pipeline
    # Delegates to library.engine.predict_submission which handles:
    # - Loading the test dataset
    # - Generating TTA views (Original, HFlip, VFlip, Rot90)
    # - Running predictions for all models
    # - Averaging probabilities
    # - Saving the final submission.csv
    predict_submission(models_config)
