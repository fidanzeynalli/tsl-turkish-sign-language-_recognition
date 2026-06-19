from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "tid_holistic_model.keras"
DATA_PATH = ROOT / "normalized_verisetim.csv"
LABELS_PATH = ROOT / "siniflar.npy"


def infer_sequence_length(feature_count: int) -> int:
    if feature_count % 258 != 0:
        raise ValueError(f"Feature count {feature_count} is not divisible by 258.")
    return feature_count // 258


def main() -> None:
    print("Loading dataset and saved label order...")
    df = pd.read_csv(DATA_PATH)
    saved_classes = np.load(LABELS_PATH, allow_pickle=True)

    X = df.drop("etiket", axis=1).values
    y = df["etiket"].astype(str).values

    sequence_length = infer_sequence_length(X.shape[1])
    print(f"Detected sequence length: {sequence_length}")

    X = X.reshape(-1, sequence_length, 258)

    # Match the exact label order that was saved during training.
    encoder = LabelEncoder()
    encoder.classes_ = saved_classes.astype(str)
    y_encoded = encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.1,
        random_state=42,
        stratify=y_encoded,
    )

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Evaluating...")
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")

    print("\nClassification report (first 20 lines):")
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    )
    for line in report.splitlines()[:20]:
        print(line)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion matrix shape: {cm.shape}")
    print(f"Test samples: {len(X_test)}")


if __name__ == "__main__":
    main()