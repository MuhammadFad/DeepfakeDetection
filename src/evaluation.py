import numpy as np
from sklearn.metrics import confusion_matrix

from .config import MODEL_DISPLAY_NAMES


def calculate_accuracy(predictions: list, ground_truths: list) -> float:
    if not predictions:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    return (correct / len(predictions)) * 100.0


def build_confusion_matrix(predictions: list, ground_truths: list) -> np.ndarray:
    """Return a 2×2 matrix with layout [[TN, FP], [FN, TP]] (Real=0, Fake=1)."""
    if not predictions:
        return np.zeros((2, 2), dtype=int)
    label_map = {'Real': 0, 'Fake': 1}
    y_true = [label_map[g] for g in ground_truths]
    y_pred = [label_map[p] for p in predictions]
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def calculate_per_class_accuracy(predictions: list, ground_truths: list) -> dict:
    real_total = real_correct = fake_total = fake_correct = 0
    for pred, truth in zip(predictions, ground_truths):
        if truth == 'Real':
            real_total += 1
            real_correct += pred == 'Real'
        else:
            fake_total += 1
            fake_correct += pred == 'Fake'
    return {
        'Real': (real_correct / real_total * 100.0) if real_total else 0.0,
        'Fake': (fake_correct / fake_total * 100.0) if fake_total else 0.0,
    }


def generate_summary_stats(all_model_results: dict) -> dict:
    summary = {}
    for model_name, results in all_model_results.items():
        valid = [(r['prediction'], r['ground_truth'])
                 for r in results if r['prediction'] is not None]
        if valid:
            preds, truths = zip(*valid)
            preds, truths = list(preds), list(truths)
        else:
            preds, truths = [], []

        cm = build_confusion_matrix(preds, truths)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        summary[model_name] = {
            'accuracy': calculate_accuracy(preds, truths),
            'per_class': calculate_per_class_accuracy(preds, truths),
            'confusion_matrix': cm,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'total': len(preds),
            'display_name': MODEL_DISPLAY_NAMES.get(model_name, model_name),
        }
    return summary
