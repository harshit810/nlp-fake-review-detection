import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self, n_jobs=-1):
        logger.info("Initializing FeatureExtractor")
        self.vader = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer(
            max_features=100,  # Reduced from 1000 for faster processing
            stop_words="english",
            max_df=0.95,  # Remove very common words
            min_df=2,  # Remove very rare words
            ngram_range=(1, 2),  # Use both unigrams and bigrams
        )
        self.n_jobs = n_jobs

    def extract_sentiment_features(self, df):
        """Extract sentiment features using VADER"""
        try:
            # Extract VADER features for each review
            features = []
            for text in df['Text']:
                vader_scores = self.vader.polarity_scores(str(text))
                features.append({
                    "vader_compound": vader_scores["compound"],
                    "vader_pos": vader_scores["pos"],
                    "vader_neg": vader_scores["neg"],
                    "vader_neu": vader_scores["neu"]
                })
            return pd.DataFrame(features)
        except Exception as e:
            logger.error("Error extracting sentiment features: %s", str(e))
            raise

    def extract_behavioral_features(self, df):
        """Extract behavioral features for each review"""
        try:
            features = []
            for _, row in df.iterrows():
                user_reviews = df[df['UserId'] == row['UserId']]
                review_count = len(user_reviews)

                if review_count > 1:
                    timestamps = pd.to_datetime(user_reviews["Time"])
                    time_span = (timestamps.max() - timestamps.min()).total_seconds()
                    review_frequency = (
                        review_count
                        if time_span == 0
                        else review_count / (time_span / 86400)
                    )
                    score_std = user_reviews["Score"].std()
                else:
                    review_frequency = 1
                    score_std = 0

                features.append({
                    "user_review_count": review_count,
                    "review_frequency": review_frequency,
                    "avg_score": user_reviews["Score"].mean(),
                    "score_std": score_std
                })
            return pd.DataFrame(features)
        except Exception as e:
            logger.error("Error extracting behavioral features: %s", str(e))
            raise

    def extract_linguistic_features(self, df):
        """Extract linguistic features from review text"""
        try:
            features = []
            for text in df['Text']:
                # Simple text statistics
                words = str(text).split()
                word_count = len(words)

                if word_count == 0:
                    features.append({
                        "word_count": 0,
                        "avg_word_length": 0,
                        "avg_sentence_length": 0
                    })
                    continue

                avg_word_length = sum(len(word) for word in words) / word_count
                sentences = str(text).split(".")
                avg_sentence_length = sum(
                    len(s.split()) for s in sentences if s.strip()
                ) / max(len([s for s in sentences if s.strip()]), 1)

                features.append({
                    "word_count": word_count,
                    "avg_word_length": avg_word_length,
                    "avg_sentence_length": avg_sentence_length
                })
            return pd.DataFrame(features)
        except Exception as e:
            logger.error("Error extracting linguistic features: %s", str(e))
            raise

    def process_review_batch(self, batch_df, user_reviews_dict):
        """Process a batch of reviews in parallel"""
        features_list = []

        for _, row in batch_df.iterrows():
            # Get user reviews from pre-computed dictionary
            user_reviews = user_reviews_dict.get(row["UserId"], pd.DataFrame())

            # Extract features
            sentiment_features = self.extract_sentiment_features(pd.DataFrame([row]))
            behavioral_features = self.extract_behavioral_features(pd.DataFrame([row]))
            linguistic_features = self.extract_linguistic_features(pd.DataFrame([row]))

            # Add metadata features
            metadata_features = {
                "score": row["Score"],
                "helpfulness_ratio": row["HelpfulnessRatio"],
            }

            # Combine features
            combined_features = pd.concat(
                [
                    sentiment_features,
                    behavioral_features,
                    linguistic_features,
                    pd.DataFrame([metadata_features])
                ],
                axis=1
            )
            features_list.append(combined_features)

        return pd.concat(features_list, ignore_index=True)

    def extract_all_features(self, df):
        """Extract all features for the dataset using parallel processing"""
        try:
            logger.info("Starting feature extraction for %d reviews", len(df))

            # Pre-compute user reviews dictionary
            logger.info("Pre-computing user reviews dictionary...")
            user_reviews_dict = {name: group for name, group in df.groupby("UserId")}

            # Compute TF-IDF features first (this is fast with scikit-learn)
            logger.info("Computing TF-IDF features...")
            tfidf_matrix = self.tfidf.fit_transform(df["Text"].astype(str))

            # Process reviews in parallel batches
            batch_size = 1000  # Adjust based on your system's memory
            batches = [df[i : i + batch_size] for i in range(0, len(df), batch_size)]

            all_features = []
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                process_batch = partial(
                    self.process_review_batch, user_reviews_dict=user_reviews_dict
                )
                for batch_features in executor.map(process_batch, batches):
                    all_features.append(batch_features)

            # Convert to DataFrame
            logger.info("Converting features to DataFrame...")
            feature_df = pd.concat(all_features, ignore_index=True)

            # Add TF-IDF features
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
            )

            final_df = pd.concat([feature_df, tfidf_df], axis=1)
            logger.info(
                "Feature extraction completed. Final feature count: %d",
                final_df.shape[1],
            )

            return final_df

        except Exception as e:
            logger.error("Error in feature extraction pipeline: %s", str(e))
            raise
