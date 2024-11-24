import pandas as pd
import numpy as np
from feature_extraction import FeatureExtractor
from model_training import FakeReviewDetector
import nltk
import warnings
import logging
import multiprocessing
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("vader_lexicon")


def load_and_preprocess_data(filepath, sample_size=10000):
    """Load and preprocess the dataset"""
    try:
        logger.info("Loading dataset from %s", filepath)

        # Load only necessary columns with specified data types
        df = pd.read_csv(
            filepath,
            usecols=["Text", "UserId", "Score", "Time", "HelpfulnessRatio"],
            dtype={
                "Text": str,
                "UserId": str,
                "Score": float,
                "HelpfulnessRatio": float,
            },
        )

        # Take a random sample for faster processing
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        logger.info("Successfully loaded %d reviews", len(df))

        # Label reviews as potentially fake based on helpfulness ratio
        df["is_fake"] = df["HelpfulnessRatio"] < 0.5
        fake_count = df["is_fake"].sum()
        logger.info("Identified %d potential fake reviews", fake_count)

        return df

    except Exception as e:
        logger.error("Error loading or preprocessing data: %s", str(e))
        raise


def evaluate_feature_sets():
    logger.info("Starting feature set evaluation...")

    # Initialize feature extractor and model trainer
    feature_extractor = FeatureExtractor()
    detector = FakeReviewDetector()

    # Load and preprocess data
    df = load_and_preprocess_data("processed_amazon_reviews.csv")

    # Extract features
    behavioral_features = feature_extractor.extract_behavioral_features(df)
    sentiment_features = feature_extractor.extract_sentiment_features(df)
    linguistic_features = feature_extractor.extract_linguistic_features(df)

    # Create DataFrames
    feature_sets = {
        "Behavioral": pd.DataFrame(behavioral_features),
        "Sentiment": pd.DataFrame(sentiment_features),
        "Linguistic": pd.DataFrame(linguistic_features),
    }

    results = {}

    # Evaluate each feature set
    for feature_name, features in feature_sets.items():
        logger.info(f"\nEvaluating {feature_name} features:")
        logger.info(f"Shape: {features.shape}")
        logger.info(f"Features: {list(features.columns)}")

        # Train and evaluate models
        model_scores = detector.train_and_evaluate(features, df["is_fake"].values)
        results[feature_name] = model_scores

    # Create comparison plots
    plot_feature_set_comparison(results)


def plot_feature_set_comparison(results):
    # Plot settings
    plt.figure(figsize=(15, 8))
    feature_sets = list(results.keys())
    models = list(results[feature_sets[0]].keys())
    metrics = ["accuracy", "precision", "recall", "f1"]

    # Create subplots for each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)

        # Prepare data for plotting
        x = np.arange(len(feature_sets))
        width = 0.25

        for j, model in enumerate(models):
            scores = [results[feat][model][metric] for feat in feature_sets]
            plt.bar(x + j * width, scores, width, label=model.replace("_", " ").title())

        plt.xlabel("Feature Sets")
        plt.ylabel(f"{metric.title()} Score")
        plt.title(f"{metric.title()} by Feature Set and Model")
        plt.xticks(x + width, feature_sets)
        plt.legend()

    plt.tight_layout()
    plt.savefig("feature_set_comparison.png")
    logger.info("Feature set comparison plot saved as 'feature_set_comparison.png")


def main():
    try:
        # Determine number of CPU cores for parallel processing
        n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        logger.info("Using %d CPU cores for parallel processing", n_jobs)

        # Load and preprocess data
        df = load_and_preprocess_data("processed_amazon_reviews.csv")
        logger.info("Dataset loaded successfully with %d reviews", len(df))

        # Extract features
        logger.info("Starting feature extraction...")
        feature_extractor = FeatureExtractor(n_jobs=n_jobs)
        X = feature_extractor.extract_all_features(df)
        logger.info("Feature extraction completed. Extracted %d features", X.shape[1])

        # Train and evaluate models
        logger.info("Starting model training and evaluation...")
        detector = FakeReviewDetector(n_jobs=n_jobs)
        results = detector.train_and_evaluate(X, df['is_fake'])

        # Print summary
        logger.info("\nModel Performance Summary:")
        for model_name, metrics in results.items():
            logger.info("\n%s Results:", model_name.upper())
            logger.info("Accuracy: %.3f", metrics["accuracy"])
            logger.info("Precision: %.3f", metrics["precision"])
            logger.info("Recall: %.3f", metrics["recall"])
            logger.info("F1 Score: %.3f", metrics["f1"])

        # Plot results
        detector.plot_results(results)
        logger.info("Performance plot saved as 'model_comparison.png'")

        # Extract feature importance from best model
        logger.info("Extracting feature importance from best model")
        feature_importance = detector.get_feature_importance(X.columns)
        if feature_importance is not None:
            logger.info("\nTop 10 Most Important Features:")
            logger.info("\n%s", feature_importance.head(10).to_string())

    except Exception as e:
        logger.error("Error in main execution: %s", str(e))
        raise


if __name__ == "__main__":
    evaluate_feature_sets()
    main()
