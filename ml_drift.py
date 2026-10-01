"""
Lightweight ML model for claim-drift detection.

The model compares an official claim with a rumor claim.

Architecture:
    Official claim + Rumor claim
              ↓
        TF-IDF features
              ↓
    Logistic Regression
              ↓
    Scope / Duration / Location / Context drift scores

This is a prototype model trained on curated examples.
It is intentionally lightweight for low-memory deployment.
"""

from __future__ import annotations

from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ----------------------------------------------------------------------------- #
# Prototype paired training data
# ----------------------------------------------------------------------------- #
#
# Format:
# (official_statement, rumor_statement, label)
#
# label = 1 → drift exists
# label = 0 → no drift
#

TRAINING_DATA = {
    "scope": [
        # Drift
        (
            "Central Service Branch is temporarily closed.",
            "All service branches in the city are closed.",
            1,
        ),
        (
            "Only the Central Branch is affected.",
            "Every branch in the city is affected.",
            1,
        ),
        (
            "The Central Service Branch is closed.",
            "All service centres are closed.",
            1,
        ),
        (
            "One branch is temporarily unavailable.",
            "All branches are unavailable.",
            1,
        ),
        (
            "The closure applies only to Central Branch.",
            "The closure applies to every branch.",
            1,
        ),

        # No drift
        (
            "Central Service Branch is temporarily closed.",
            "Central Service Branch is temporarily closed.",
            0,
        ),
        (
            "Only the Central Branch is affected.",
            "Only the Central Branch is affected.",
            0,
        ),
        (
            "The Central Service Branch is closed.",
            "The Central Service Branch is closed.",
            0,
        ),
        (
            "One branch is temporarily unavailable.",
            "One branch is temporarily unavailable.",
            0,
        ),
        (
            "The closure applies only to Central Branch.",
            "The closure applies only to Central Branch.",
            0,
        ),
    ],

    "duration": [
        # Drift
        (
            "Central Branch is temporarily closed until 12 PM.",
            "Central Branch is closed indefinitely.",
            1,
        ),
        (
            "The branch will reopen at 12 PM.",
            "The branch will remain closed permanently.",
            1,
        ),
        (
            "The closure lasts from 8 AM to 12 PM.",
            "The closure is until further notice.",
            1,
        ),
        (
            "The service interruption is temporary.",
            "The service interruption is permanent.",
            1,
        ),
        (
            "The branch will reopen after maintenance.",
            "The branch will never reopen.",
            1,
        ),

        # No drift
        (
            "Central Branch is temporarily closed until 12 PM.",
            "Central Branch is temporarily closed until 12 PM.",
            0,
        ),
        (
            "The branch will reopen at 12 PM.",
            "The branch will reopen at 12 PM.",
            0,
        ),
        (
            "The closure lasts from 8 AM to 12 PM.",
            "The closure lasts from 8 AM to 12 PM.",
            0,
        ),
        (
            "The service interruption is temporary.",
            "The service interruption is temporary.",
            0,
        ),
        (
            "The branch will reopen after maintenance.",
            "The branch will reopen after maintenance.",
            0,
        ),
    ],

    "location": [
        # Drift
        (
            "Central Service Branch is temporarily closed.",
            "All branches across the city are closed.",
            1,
        ),
        (
            "The Central Branch is affected.",
            "Every location in the city is affected.",
            1,
        ),
        (
            "Only Central Branch is unavailable.",
            "All city locations are unavailable.",
            1,
        ),
        (
            "The closure applies to Central Branch.",
            "The closure applies citywide.",
            1,
        ),
        (
            "One service location is closed.",
            "All service locations are closed.",
            1,
        ),

        # No drift
        (
            "Central Service Branch is temporarily closed.",
            "Central Service Branch is temporarily closed.",
            0,
        ),
        (
            "The Central Branch is affected.",
            "The Central Branch is affected.",
            0,
        ),
        (
            "Only Central Branch is unavailable.",
            "Only Central Branch is unavailable.",
            0,
        ),
        (
            "The closure applies to Central Branch.",
            "The closure applies to Central Branch.",
            0,
        ),
        (
            "One service location is closed.",
            "One service location is closed.",
            0,
        ),
    ],

    "context": [
        # Drift
        (
            "Central Branch is temporarily closed for maintenance and will reopen at 12 PM.",
            "Central Branch is closed.",
            1,
        ),
        (
            "The branch is closed from 8 AM to 12 PM due to maintenance.",
            "The branch is closed.",
            1,
        ),
        (
            "The branch will reopen at 12 PM.",
            "The branch is closed.",
            1,
        ),
        (
            "Other branches remain operational.",
            "The branches are closed.",
            1,
        ),
        (
            "Residents can use other branches before 12 PM.",
            "Residents cannot get service anywhere.",
            1,
        ),

        # No drift
        (
            "Central Branch is temporarily closed for maintenance and will reopen at 12 PM.",
            "Central Branch is temporarily closed for maintenance and will reopen at 12 PM.",
            0,
        ),
        (
            "The branch is closed from 8 AM to 12 PM due to maintenance.",
            "The branch is closed from 8 AM to 12 PM due to maintenance.",
            0,
        ),
        (
            "The branch will reopen at 12 PM.",
            "The branch will reopen at 12 PM.",
            0,
        ),
        (
            "Other branches remain operational.",
            "Other branches remain operational.",
            0,
        ),
        (
            "Residents can use other branches before 12 PM.",
            "Residents can use other branches before 12 PM.",
            0,
        ),
    ],
}


class DriftClassifier:
    """Small binary TF-IDF + Logistic Regression classifier."""

    def __init__(self, examples):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=1000,
            sublinear_tf=True,
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )

        texts = []
        labels = []

        for official, rumor, label in examples:
            texts.append(self._pair_text(official, rumor))
            labels.append(label)

        features = self.vectorizer.fit_transform(texts)
        self.model.fit(features, labels)

    @staticmethod
    def _pair_text(official: str, rumor: str) -> str:
        """
        Represent the relationship between the official statement
        and the rumor as a single ML input.
        """
        return (
            f"OFFICIAL_CLAIM {official} "
            f"RUMOR_CLAIM {rumor}"
        )

    def predict(
        self,
        official: str,
        rumor: str,
    ) -> float:
        """Return probability that this drift type exists."""

        if not official or not rumor:
            return 0.0

        pair = self._pair_text(official, rumor)
        features = self.vectorizer.transform([pair])

        probabilities = self.model.predict_proba(features)[0]

        # Probability of class 1 = drift.
        class_to_probability = dict(
            zip(
                self.model.classes_,
                probabilities,
            )
        )

        return round(
            float(class_to_probability.get(1, 0.0)),
            4,
        )


# ----------------------------------------------------------------------------- #
# Build the four lightweight classifiers once at startup.
# ----------------------------------------------------------------------------- #

_MODELS: Dict[str, DriftClassifier] = {
    category: DriftClassifier(examples)
    for category, examples in TRAINING_DATA.items()
}


def predict_drift(
    official_claim: str,
    rumor_claim: str,
) -> Dict[str, float]:
    """
    Compare an official claim with a rumor and return
    independent drift probabilities.
    """

    scores = {
        category: model.predict(
            official_claim,
            rumor_claim,
        )
        for category, model in _MODELS.items()
    }

    scores["overall"] = round(
        max(scores.values()),
        4,
    )

    return scores
