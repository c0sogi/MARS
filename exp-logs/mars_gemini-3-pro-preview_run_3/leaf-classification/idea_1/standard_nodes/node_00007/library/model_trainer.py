import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss
from library.utils import seed_everything, create_submission_file
from library.data_loader import LeafDataManager


class EnsembleTrainer:
    def __init__(self, data_manager=None):
        """
        Initializes the EnsembleTrainer.

        Args:
            data_manager (LeafDataManager, optional): An instance of LeafDataManager.
                                                      If None, a new instance is created.
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
        self.classes = self.data_manager.get_classes()

    def grid_search_regularization(self, c_values=None):
        """
        Performs a grid search to find the optimal regularization strength (C).

        Args:
            c_values (list or np.ndarray, optional): Array of C values to test.
                                                     Defaults to a logspace range.

        Returns:
            float: The best C value found.
        """
        X_train, y_train = self.data_manager.get_train_data()
        X_val, y_val = self.data_manager.get_val_data()

        if c_values is None:
            # Logarithmic scale from 1e-3 to 1e3 covers a wide range of regularization strengths
            c_values = np.logspace(-3, 3, 13)

        print(f"Starting grid search over C values: {c_values}")

        best_loss = float("inf")
        best_c = c_values[0]

        for c in c_values:
            # Initialize Logistic Regression with L-BFGS solver for multinomial loss
            # C is the inverse of regularization strength (smaller C = stronger regularization)
            clf = LogisticRegression(
                C=c,
                solver="lbfgs",
                multi_class="multinomial",
                max_iter=1000,  # Sufficient iterations for convergence
                random_state=42,
                n_jobs=-1,
            )

            clf.fit(X_train, y_train)

            # Predict probabilities on validation set
            y_val_prob = clf.predict_proba(X_val)

            # Calculate Log Loss
            loss = log_loss(y_val, y_val_prob)

            print(f"C={c}: Validation Log Loss = {loss}")

            if loss < best_loss:
                best_loss = loss
                best_c = c

        print(f"Grid search complete. Best C: {best_c} with Log Loss: {best_loss}")
        self.best_c = best_c
        return best_c

    def train(self, c=None):
        """
        Trains the final ensemble (LR + LDA).

        Args:
            c (float, optional): Regularization strength for LR. If None, uses the best_c from grid search.
        """
        if c is None:
            if self.best_c is None:
                print("No C provided. Running grid search with default values...")
                self.grid_search_regularization()
                c = self.best_c
            else:
                c = self.best_c

        print(f"Training final ensemble with LR(C={c}) and LDA(shrinkage='auto')...")

        X_train, y_train = self.data_manager.get_train_data()

        # Train Logistic Regression
        self.lr_model = LogisticRegression(
            C=c,
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )
        self.lr_model.fit(X_train, y_train)

        # Train LDA with shrinkage (Cite solution_lesson_node_00001: Linear models effective for high-dim/low-sample)
        self.lda_model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.lda_model.fit(X_train, y_train)

        # Calculate and print training metrics
        train_probs = self.predict_proba(X_train)
        train_loss = log_loss(y_train, train_probs)
        print(f"Final ensemble training Log Loss: {train_loss}")

        # Calculate and print validation metrics for confirmation
        X_val, y_val = self.data_manager.get_val_data()
        val_probs = self.predict_proba(X_val)
        val_loss = log_loss(y_val, val_probs)
        print(f"Final ensemble validation Log Loss: {val_loss}")

    def predict_proba(self, X):
        """
        Generates averaged probability predictions from LR and LDA.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        if self.lr_model is None or self.lda_model is None:
            raise ValueError("Models have not been trained yet. Call train() first.")

        p_lr = self.lr_model.predict_proba(X)
        p_lda = self.lda_model.predict_proba(X)

        # Simple averaging ensemble
        return 0.5 * p_lr + 0.5 * p_lda

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
