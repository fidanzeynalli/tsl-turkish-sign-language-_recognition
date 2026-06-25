from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "tid_holistic_model.keras"
DATASET_CANDIDATES = [ROOT / "final_verisetim.csv", ROOT / "normalized_verisetim.csv"]
LABELS_PATH = ROOT / "siniflar.npy"


def infer_sequence_length(feature_count: int) -> int:
    if feature_count % 258 != 0:
        raise ValueError(f"Feature count {feature_count} is not divisible by 258.")
    return feature_count // 258


def main() -> None:
    print("Loading dataset and saved label order...")
    saved_classes = np.load(LABELS_PATH, allow_pickle=True)

    data_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("No evaluation dataset found. Expected final_verisetim.csv or normalized_verisetim.csv.")

    print(f"Using dataset: {data_path.name}")
    df = pd.read_csv(data_path)

    X = df.drop("etiket", axis=1).values
    y = df["etiket"].astype(str).values

    sequence_length = infer_sequence_length(X.shape[1])
    print(f"Detected sequence length: {sequence_length}")

    X = X.reshape(-1, sequence_length, 258)

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Evaluating...")
    probabilities = model.predict(X, verbose=0)
    y_pred_indices = np.argmax(probabilities, axis=1)
    predicted_labels = saved_classes.astype(str)[y_pred_indices]

    allowed_labels = set(saved_classes.astype(str))
    correct_predictions = 0
    skipped_unseen = 0

    for true_label, predicted_label in zip(y, predicted_labels):
        if true_label not in allowed_labels:
            skipped_unseen += 1
            continue
        if true_label == predicted_label:
            correct_predictions += 1

    total_samples = len(y)
    overall_accuracy = correct_predictions / total_samples

    print(f"Overall accuracy over all labels: {overall_accuracy * 100:.2f}%")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Total samples: {total_samples}")
    print(f"Unseen labels counted as incorrect: {skipped_unseen}")

    seen_mask = np.array([label in allowed_labels for label in y])
    seen_true = y[seen_mask]
    seen_pred = predicted_labels[seen_mask]
    if len(seen_true) > 0:
        seen_acc = accuracy_score(seen_true, seen_pred)
        print(f"Accuracy on labels known to the model: {seen_acc * 100:.2f}%")

        print("\nClassification report (first 20 lines):")
        report = classification_report(
            seen_true,
            seen_pred,
            labels=saved_classes.astype(str),
            target_names=saved_classes.astype(str),
            zero_division=0,
        )
        for line in report.splitlines()[:20]:
            print(line)

        cm = confusion_matrix(seen_true, seen_pred, labels=saved_classes.astype(str))
        print(f"\nConfusion matrix shape: {cm.shape}")


if __name__ == "__main__":
    main()