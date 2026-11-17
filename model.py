"""Train and inspect a small nearest-neighbor classifier."""
from collections import Counter

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

PIXELS = 64
NEIGHBORS = 5


class DigitModel:
    def __init__(self):
        digits = load_digits()
        self.images, test_images, self.labels, test_labels = train_test_split(
            digits.data, digits.target, test_size=0.25, stratify=digits.target,
            random_state=42,
        )
        self.classifier = KNeighborsClassifier(n_neighbors=NEIGHBORS)
        self.classifier.fit(self.images, self.labels)
        self.metrics = {
            "train_count": len(self.images),
            "test_count": len(test_images),
            "accuracy": float(accuracy_score(test_labels, self.classifier.predict(test_images))),
        }

    def explain(self, pixels):
        if (
            not isinstance(pixels, list)
            or len(pixels) != PIXELS
            or any(type(value) is not int or not 0 <= value <= 16 for value in pixels)
        ):
            raise ValueError("Provide 64 integer pixels between 0 and 16.")
        if not any(pixels):
            raise ValueError("Draw a digit or load the sample first.")

        distances, indices = self.classifier.kneighbors([pixels])
        labels = self.labels[indices[0]].tolist()
        votes = Counter(labels)
        # Uniform voting with the lowest digit winning ties, as in sorted class order.
        digit = min(votes, key=lambda label: (-votes[label], label))
        return {
            "digit": digit,
            "votes": dict(sorted(votes.items())),
            "neighbors": [
                {"label": label, "distance": round(float(distance), 2),
                 "pixels": self.images[index].astype(int).tolist()}
                for label, distance, index in zip(labels, distances[0], indices[0])
            ],
        }


if __name__ == "__main__":
    print(DigitModel().metrics)
