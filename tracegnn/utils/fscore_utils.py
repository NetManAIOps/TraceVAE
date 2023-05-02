from typing import *

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

__all__ = [
    'fscore_for_precision_and_recall',
    'best_fscore',
    'auc_score',
]


def fscore_for_precision_and_recall(precision: np.ndarray,
                                    recall: np.ndarray) -> np.ndarray:
    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)
    return np.where(
        (precision == 0) | (recall == 0),
        0.0,
        2. * np.exp(
            np.log(np.maximum(precision, 1e-8)) +
            np.log(np.maximum(recall, 1e-8)) -
            np.log(np.maximum(precision + recall, 1e-8))
        )
    )


def best_fscore(proba: np.ndarray,
                truth: np.ndarray) -> Tuple[float, float]:
    precision, recall, threshold = precision_recall_curve(truth, proba)
    fscore = fscore_for_precision_and_recall(precision, recall)
    idx = np.argmax(fscore[:-1])
    return fscore[idx], threshold[idx], precision[idx], recall[idx]


def auc_score(proba: np.ndarray, truth: np.ndarray) -> float:
    return float(average_precision_score(truth, proba))
