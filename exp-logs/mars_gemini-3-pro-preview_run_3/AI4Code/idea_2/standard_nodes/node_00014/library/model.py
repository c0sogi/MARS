import lightgbm as lgb
import joblib
from library.config import Config


class LGBMModel:
    """
    Wrapper for LightGBM Regressor.
    Cite solution_lesson_node_00012: Explicit Alignment Features Outperform End-to-End Attention.
    """

    def __init__(self):
        self.model = None

    def train(self, train_data, valid_data):
        self.model = lgb.train(
            Config.LGBM_PARAMS,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=[lgb.log_evaluation(100)],
        )

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
