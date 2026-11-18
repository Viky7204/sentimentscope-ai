"""
Train Sentiment Analysis model for SentimentScope AI.
Run this once before starting the app.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from preprocess import full_clean


def create_sample_dataset():
    """
    Small balanced dataset so the project works immediately.
    Labels: 0 = Negative, 1 = Neutral, 2 = Positive
    """
    data = {
        "text": [
            # Positive
            "This product is amazing! Totally worth the money. Highly recommended.",
            "Excellent quality and fast delivery. I am very happy with my purchase.",
            "Best phone I have ever used. Battery life is outstanding.",
            "Absolutely love this! Great design and works perfectly.",
            "Fantastic experience. Customer support was also very helpful.",
            "Wonderful product. Exceeded my expectations in every way.",
            "Very satisfied. Will definitely buy again from this brand.",
            "Outstanding performance and beautiful packaging.",
            "I am extremely happy with this purchase. Five stars!",
            "Great value for money. Performs better than expected.",
            "Superb quality. My family loves it too.",
            "Impressive features and very easy to use.",

            # Negative
            "Worst product ever. Completely useless and waste of money.",
            "Very disappointed. It stopped working after two days.",
            "Poor quality. Looks cheap and feels fragile.",
            "Terrible experience. Would not recommend to anyone.",
            "Not worth the price at all. Very bad performance.",
            "I regret buying this. Customer service was also rude.",
            "Defective item received. Extremely frustrated.",
            "Horrible build quality. Broke within a week.",
            "Do not buy this. Complete scam and low quality.",
            "Awful product. Nothing works as advertised.",
            "Very poor experience. Returning it immediately.",
            "Cheap material and bad finishing. Not happy.",

            # Neutral
            "It is okay. Nothing special but does the job.",
            "Average product. Expected a bit more for the price.",
            "It's fine for basic use. Not great, not terrible.",
            "Decent quality. Meets minimum expectations.",
            "The product is alright. Delivery was on time.",
            "Neither good nor bad. Just an average experience.",
            "Works as expected. No major complaints or praise.",
            "It's acceptable. Might buy again if needed.",
            "Standard quality. Nothing stands out particularly.",
            "Okay product for the price range.",
            "Satisfied to some extent. Could be better.",
            "Fair enough. Does what it claims to do.",
        ],
        "label": [
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   # Positive
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   # Negative
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   # Neutral
        ]
    }
    return pd.DataFrame(data)


def train_model():
    print("Creating sample dataset...")
    df = create_sample_dataset()

    print("Cleaning text...")
    df["cleaned"] = df["text"].apply(full_clean)

    X = df["cleaned"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training TF-IDF + Logistic Regression...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, multi_class="auto"))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Negative", "Neutral", "Positive"]
    ))

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/sentiment_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved → {model_path}")

    # Save sample data
    os.makedirs("data", exist_ok=True)
    df[["text", "label"]].to_csv("data/sample_reviews.csv", index=False)
    print("Sample data saved → data/sample_reviews.csv")

    return pipeline


if __name__ == "__main__":
    train_model()
    print("\n✅ Training completed successfully!")
