import sys
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef

# Import from provided libraries
from library.config import ProjectConfig
from library.utils import seed_everything, get_logger
from library.data_pipeline import DataPipeline
from library.model_trainer import DualStreamTrainer
from library.metric_optimizer import MetricOptimizer


def analyze_failure_modes(X, y_true, probs, stream_name, logger):
    """
    Performs failure analysis by correlating error magnitude with features.
    """
    if X.empty:
        return

    # Calculate absolute error
    errors = np.abs(y_true - probs)

    # Create a temporary dataframe for correlation analysis
    # We use a subset or the full set depending on size, but X is validation set so it fits in memory
    df_analysis = X.copy()
    df_analysis["error_magnitude"] = errors

    # Calculate correlation with error_magnitude
    # Drop columns that are constant or all NaN to avoid warnings/errors
    df_analysis = df_analysis.loc[:, df_analysis.nunique() > 1]

    if "error_magnitude" not in df_analysis.columns:
        return

    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    # Get top 5 features most correlated with error (positive or negative)
    top_corrs = correlations.abs().sort_values(ascending=False).head(5)

    logger.info(f"--- Failure Analysis: {stream_name} ---")
    logger.info("Top 5 features correlated with prediction error:")
    logger.info(f"\n{top_corrs}")


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    seed_everything(ProjectConfig.SEED)
    logger = get_logger("RunFile")

    logger.info(
        "Starting Physically-Consistent Hybrid-Context Dual-Stream GBDT Pipeline"
    )

    # Fast Baseline Overrides
    # Limit estimators to ensure quick execution within time limits
    ProjectConfig.XGB_PARAMS_STREAM_A["n_estimators"] = 5000
    ProjectConfig.XGB_PARAMS_STREAM_B["n_estimators"] = 5000
    # Ensure GPU usage is explicit (though config defaults to True)
    ProjectConfig.USE_GPU = True

    # 2. Data Loading and Processing
    pipeline = DataPipeline()

    # Load Training Data (with undersampling applied internally)
    logger.info("Loading Training Data...")
    data_train = pipeline.run(mode="train", load_cached_data=True)

    # Load Validation Data (Full set, no undersampling)
    logger.info("Loading Validation Data...")
    data_val = pipeline.run(mode="validation", load_cached_data=True)

    # 3. Model Training
    trainer = DualStreamTrainer()

    # Train Stream A (Interaction Model)
    logger.info("Training Stream A (Player-Player)...")
    trainer.train_stream(
        X_train=data_train["stream_a"]["X"],
        y_train=data_train["stream_a"]["y"],
        X_val=data_val["stream_a"]["X"],
        y_val=data_val["stream_a"]["y"],
        stream_type="A",
    )

    # Train Stream B (Impact Model)
    logger.info("Training Stream B (Player-Ground)...")
    trainer.train_stream(
        X_train=data_train["stream_b"]["X"],
        y_train=data_train["stream_b"]["y"],
        X_val=data_val["stream_b"]["X"],
        y_val=data_val["stream_b"]["y"],
        stream_type="B",
    )

    # 4. Validation and Threshold Optimization
    optimizer = MetricOptimizer()

    # Inference on Validation Set
    logger.info("Running Inference on Validation Set...")
    probs_val_a = trainer.predict_stream(trainer.model_a, data_val["stream_a"]["X"])
    probs_val_b = trainer.predict_stream(trainer.model_b, data_val["stream_b"]["X"])

    # Optimize Thresholds
    thresh_a = optimizer.find_optimal_threshold(
        data_val["stream_a"]["y"], probs_val_a, "A"
    )
    thresh_b = optimizer.find_optimal_threshold(
        data_val["stream_b"]["y"], probs_val_b, "B"
    )

    # Calculate Final Validation Metric (Global MCC)
    # Combine predictions from both streams
    preds_val_a = (probs_val_a >= thresh_a).astype(int)
    preds_val_b = (probs_val_b >= thresh_b).astype(int)

    # Concatenate targets and predictions
    y_true_global = pd.concat(
        [data_val["stream_a"]["y"], data_val["stream_b"]["y"]], axis=0
    )
    y_pred_global = np.concatenate([preds_val_a, preds_val_b])

    final_mcc = matthews_corrcoef(y_true_global, y_pred_global)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")
    analyze_failure_modes(
        data_val["stream_a"]["X"],
        data_val["stream_a"]["y"],
        probs_val_a,
        "Stream A",
        logger,
    )
    analyze_failure_modes(
        data_val["stream_b"]["X"],
        data_val["stream_b"]["y"],
        probs_val_b,
        "Stream B",
        logger,
    )

    # 6. Submission Generation
    THRESHOLD_SCORE = 0.7008

    if final_mcc > THRESHOLD_SCORE:
        logger.info(
            f"Validation MCC ({final_mcc}) exceeds threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Load Test Data
        logger.info("Loading Test Data...")
        data_test = pipeline.run(mode="test", load_cached_data=True)

        # Inference on Test Set
        probs_test_a = trainer.predict_stream(
            trainer.model_a, data_test["stream_a"]["X"]
        )
        probs_test_b = trainer.predict_stream(
            trainer.model_b, data_test["stream_b"]["X"]
        )

        # Generate Submission File
        thresholds = {"A": thresh_a, "B": thresh_b}
        optimizer.generate_submission(
            probs_a=probs_test_a,
            ids_a=data_test["stream_a"]["ids"],
            probs_b=probs_test_b,
            ids_b=data_test["stream_b"]["ids"],
            thresholds=thresholds,
        )
    else:
        logger.warning(
            f"Validation MCC ({final_mcc}) did not meet threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
