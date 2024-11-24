import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class FakeReviewDetector:
    def __init__(self, n_jobs=-1):
        logger.info("Initializing FakeReviewDetector")
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=n_jobs),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,  # Limit tree depth for faster training
                min_samples_split=20,  # Require more samples to split
                min_samples_leaf=10,  # Require more samples in leaves
                n_jobs=n_jobs,
                random_state=42,
            ),
            "svm": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                cache_size=2000,  # Increase cache size for faster training
                probability=True,
                random_state=42,
            ),
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.n_jobs = n_jobs

    def train_and_evaluate_model(self, name, model, X_train, X_test, y_train, y_test):
        """Train and evaluate a single model"""
        try:
            logger.info("Training %s...", name)

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
            }

            # Perform cross-validation
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring="f1",
                n_jobs=self.n_jobs,  # Parallel cross-validation
            )
            metrics["cv_f1_mean"] = cv_scores.mean()
            metrics["cv_f1_std"] = cv_scores.std()

            return name, model, metrics

        except Exception as e:
            logger.error("Error training %s: %s", name, str(e))
            raise

    def train_and_evaluate(self, X, y):
        """Train and evaluate all models, return detailed metrics for each."""
        logger.info("Starting model training and evaluation")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        best_f1 = 0
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            logger.info(f"\nTraining and evaluating {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            # Store results
            results[model_name] = metrics
            
            # Log results
            logger.info(f"{model_name} Results:")
            for metric, value in metrics.items():
                logger.info(f"{metric.title()}: {value:.3f}")
            
            # Update best model
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                self.best_model = model_name
                logger.info(f"New best model: {model_name} (F1={best_f1:.3f})")
        
        return results

    def plot_results(self, results):
        """Plot model comparison results"""
        try:
            logger.info("Plotting model comparison results")

            # Prepare data for plotting
            models = list(results.keys())
            metrics = ["accuracy", "precision", "recall", "f1"]

            # Create figure
            plt.figure(figsize=(10, 5))
            x = np.arange(len(models))
            width = 0.2

            # Plot bars for each metric
            for i, metric in enumerate(metrics):
                values = [results[model][metric] for model in models]
                plt.bar(x + i * width, values, width, label=metric.capitalize())

            plt.xlabel("Models")
            plt.ylabel("Score")
            plt.title("Model Performance Comparison")
            plt.xticks(x + width * 1.5, models)
            plt.legend()

            # Save plot
            plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Plot saved as 'model_comparison.png'")

        except Exception as e:
            logger.error("Error plotting results: %s", str(e))
            raise

    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model"""
        try:
            if self.best_model is None:
                logger.warning("No best model available. Train models first.")
                return None

            logger.info("Extracting feature importance from best model")
            model = self.models[self.best_model]  # Get the actual model object

            if isinstance(model, RandomForestClassifier):
                importance = model.feature_importances_
            elif isinstance(model, LogisticRegression):
                importance = np.abs(model.coef_[0])
            else:
                logger.warning(
                    "Feature importance not available for %s",
                    type(model).__name__,
                )
                return None

            # Create feature importance DataFrame
            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": importance}
            ).sort_values("importance", ascending=False)

            logger.info("Feature importance extracted successfully")
            return feature_importance

        except Exception as e:
            logger.error("Error getting feature importance: %s", str(e))
            raise

    def prepare_data(self, X, y):
        """Prepare data for training"""
        logger.info("Preparing data for training")
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def predict(self, X):
        """Make predictions using the best model"""
        try:
            if self.best_model is None:
                logger.warning("No best model available. Train models first.")
                return None

            logger.info("Making predictions using the best model")

            X_scaled = self.scaler.transform(X)
            return self.models[self.best_model].predict(X_scaled)

        except Exception as e:
            logger.error("Error making predictions: %s", str(e))
            raise
