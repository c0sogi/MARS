import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
from library.utils import seed_everything, create_submission_file
from library.data_loader import LeafDataManager


class EnsembleTrainer:
    def __init__(self, data_manager=None):
        """
        Initializes the EnsembleTrainer with Logistic Regression and LDA.
        Cite solution_lesson_node_00001: Prioritizing linear models for high-dimensional, low-sample data.
        """
        seed_everything()
        if data_manager is None:
            self.data_manager = LeafDataManager()
            self.data_manager.process_data(load_cached_data=True)
        else:
            self.data_manager = data_manager

        self.lr_model = None
        self.lda_model = None
        self.best_c = None
        self.best_alpha = 0.5  # Default weight for LR
        self.classes = self.data_manager.get_classes()

    def grid_search_regularization(self, c_values=None):
        """
        Performs a grid search to find the optimal C for Logistic Regression.
        """
        X_train, y_train = self.data_manager.get_train_data()
        X_val, y_val = self.data_manager.get_val_data()

        if c_values is None:
            # Refined search range based on previous best of ~31.6
            c_values = np.logspace(0, 3, 20)

        print(f"Starting grid search over C values: {c_values}")

        best_loss = float("inf")
        best_c = c_values[0]

        for c in c_values:
            clf = LogisticRegression(
                C=c,
                solver="lbfgs",
                multi_class="multinomial",
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            y_val_prob = clf.predict_proba(X_val)
            loss = log_loss(y_val, y_val_prob)

            if loss < best_loss:
                best_loss = loss
                best_c = c

        print(f"Grid search complete. Best C: {best_c} with Log Loss: {best_loss}")
        self.best_c = best_c
        return best_c

    def train(self, c=None):
        """
        Trains both LR and LDA, then optimizes their ensemble weight.
        """
        if c is None:
            if self.best_c is None:
                self.grid_search_regularization()
                c = self.best_c
            else:
                c = self.best_c

        print(f"Training Ensemble (LR C={c}, LDA shrinkage='auto')...")

        X_train, y_train = self.data_manager.get_train_data()
        X_val, y_val = self.data_manager.get_val_data()

        # 1. Train Logistic Regression
        self.lr_model = LogisticRegression(
            C=c,
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )
        self.lr_model.fit(X_train, y_train)
        lr_val_probs = self.lr_model.predict_proba(X_val)
        lr_loss = log_loss(y_val, lr_val_probs)
        print(f"LR Validation Log Loss: {lr_loss}")

        # 2. Train LDA
        # Shrinkage 'auto' handles high dimensionality and multicollinearity well
        self.lda_model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.lda_model.fit(X_train, y_train)
        lda_val_probs = self.lda_model.predict_proba(X_val)
        lda_loss = log_loss(y_val, lda_val_probs)
        print(f"LDA Validation Log Loss: {lda_loss}")

        # 3. Optimize Ensemble Weight
        # P = alpha * LR + (1-alpha) * LDA
        def loss_func(alpha):
            p_ens = alpha * lr_val_probs + (1 - alpha) * lda_val_probs
            return log_loss(y_val, p_ens)

        res = minimize_scalar(loss_func, bounds=(0, 1), method="bounded")
        self.best_alpha = res.x
        ens_loss = res.fun

        print(f"Optimal LR Weight: {self.best_alpha:.4f}")
        print(f"Ensemble Validation Log Loss: {ens_loss}")

    def predict_proba(self, X):
        """
        Generates weighted probability predictions.
        """
        if self.lr_model is None or self.lda_model is None:
            raise ValueError("Models have not been trained yet.")

        lr_probs = self.lr_model.predict_proba(X)
        lda_probs = self.lda_model.predict_proba(X)
        return self.best_alpha * lr_probs + (1 - self.best_alpha) * lda_probs

    def generate_submission(self, output_path="./submission/submission.csv"):
        """
        Generates predictions for the test set and saves them to a CSV file.

        Args:
            output_path (str): Path to save the submission file.
        """
        print("Generating submission file...")
        X_test, test_ids = self.data_manager.get_test_data()

        probs = self.predict_proba(X_test)

        create_submission_file(
            ids=test_ids, probs=probs, class_names=self.classes, output_path=output_path
        )
        print(f"Submission saved to {output_path}")
