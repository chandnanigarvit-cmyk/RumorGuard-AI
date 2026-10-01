"""
Lightweight ML model for claim-drift detection.

Uses TF-IDF + Logistic Regression.
The model is intentionally small so it can run on a
512 MB Render instance.
"""

from __future__ import annotations

import re
from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ----------------------------------------------------------------------------- #
# Curated prototype training data
# ----------------------------------------------------------------------------- #

TRAINING_DATA = [
    # NORMAL / NO DRIFT
    ("Central branch is temporarily closed until 12 PM.", "normal"),
    ("The Central Service Branch will reopen at 12 PM.", "normal"),
    ("Central branch is closed from 8 AM to 12 PM.", "normal"),
    ("Other service branches remain operational.", "normal"),
    ("The branch is temporarily unavailable due to maintenance.", "normal"),

    # SCOPE DRIFT
    ("All branches are closed.", "scope"),
    ("Every service centre in the city is closed.", "scope"),
    ("All service centres are unavailable.", "scope"),
    ("The entire city has no service centres operating.", "scope"),
    ("Every branch has been shut down.", "scope"),

    # DURATION DRIFT
    ("The branch is closed indefinitely.", "duration"),
    ("The service centre will remain closed permanently.", "duration"),
    ("The branch will not reopen.", "duration"),
    ("The service centre is closed until further notice.", "duration"),
    ("The closure is permanent.", "duration"),

    # LOCATION DRIFT
    ("Branches across the city are closed.", "location"),
    ("The whole city is affected by the closure.", "location"),
    ("All locations in the city are unavailable.", "location"),
    ("Every location has been affected.", "location"),
    ("The citywide service is unavailable.", "location"),

    # CONTEXT DRIFT
    ("The branch is closed.", "context"),
    ("The service centre is unavailable.", "context"),
    ("The branch stopped service.", "context"),
    ("The centre is not operating.", "context"),
    ("Service has been stopped at the branch.", "context"),
]


class ClaimDriftModel:
    """Small TF-IDF + Logistic Regression classifier."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=500,
            sublinear_tf=True,
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )

        self._train()

    def _train(self) -> None:
        texts = [item[0] for item in TRAINING_DATA]
        labels = [item[1] for item in TRAINING_DATA]

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, text: str) -> Dict[str, float | str]:
        """Return the dominant drift category and probability."""

        if not text or not text.strip():
            return {
                "category": "normal",
                "confidence": 0.0,
            }

        X = self.vectorizer.transform([text])

        probabilities = self.model.predict_proba(X)[0]
        classes = self.model.classes_

        best_index = probabilities.argmax()

        return {
            "category": str(classes[best_index]),
            "confidence": round(
                float(probabilities[best_index]),
                4,
            ),
        }


# Load once when the application starts.
_model = ClaimDriftModel()


def predict_drift(text: str) -> Dict[str, float | str]:
    """Public prediction function."""
    return _model.predict(text)
