from lightgbm import LGBMClassifier
from library.config import SEED


def get_lgbm_classifier(
    num_leaves: int = 15,
    reg_alpha: float = 1.0,
    min_child_samples: int = 20,
    learning_rate: float = 0.05,
    n_estimators: int = 200,
) -> LGBMClassifier:
    """
    Initializes a LightGBM Classifier with constraints to prevent overfitting.
    """
    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=num_leaves,
        reg_alpha=reg_alpha,
        min_child_samples=min_child_samples,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        # Feature subsampling to reduce reliance on specific embedding dimensions
        colsample_bytree=0.5,
        subsample=0.8,
        subsample_freq=1,
    )

    return model
