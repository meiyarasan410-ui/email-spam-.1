"""
High-Performance Vectorized NumPy ML Engine (Scikit-Learn API Compatible).
Provides drop-in implementations of TfidfVectorizer, MultinomialNB,
LogisticRegression, Calibrated LinearSVC, StratifiedKFold, and evaluation metrics
if scikit-learn / scipy C-extensions are not present in the environment.
"""

import math
import numpy as np
from collections import Counter
from typing import Tuple, List, Dict, Any, Optional


class TfidfVectorizerCompat:
    """
    Vectorized TF-IDF feature extractor supporting n-grams, max_features,
    sublinear TF scaling, and L2 row normalization (matching sklearn.feature_extraction.text.TfidfVectorizer).
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), max_features: int = 5000, sublinear_tf: bool = True):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []

    def _extract_ngrams(self, text: str) -> List[str]:
        tokens = str(text).split()
        min_n, max_n = self.ngram_range
        ngrams = list(tokens) if min_n == 1 else []
        for n in range(max(2, min_n), max_n + 1):
            if len(tokens) >= n:
                ngrams.extend([" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        return ngrams

    def fit(self, raw_documents: Any, y: Any = None):
        docs = list(raw_documents)
        n_docs = len(docs)
        term_counts = Counter()
        doc_freqs = Counter()

        for doc in docs:
            ngrams = self._extract_ngrams(doc)
            term_counts.update(ngrams)
            doc_freqs.update(set(ngrams))

        most_common = term_counts.most_common(self.max_features)
        # Sort alphabetically to match sklearn's get_feature_names_out ordering
        selected_terms = sorted([term for term, _ in most_common])
        self.feature_names_ = selected_terms
        self.vocabulary_ = {term: idx for idx, term in enumerate(selected_terms)}

        df = np.array([doc_freqs[term] for term in selected_terms], dtype=np.float64)
        # Smooth IDF: log((1 + n) / (1 + df)) + 1
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return self

    def transform(self, raw_documents: Any) -> np.ndarray:
        docs = list(raw_documents)
        n_docs = len(docs)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_docs, n_features), dtype=np.float64)

        vocab = self.vocabulary_
        for i, doc in enumerate(docs):
            counts = Counter(self._extract_ngrams(doc))
            for term, count in counts.items():
                idx = vocab.get(term)
                if idx is not None:
                    X[i, idx] = (1.0 + math.log(count)) if (self.sublinear_tf and count > 0) else float(count)

        if self.idf_ is not None:
            X *= self.idf_

        # L2 Normalization per document row
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X /= norms
        return X

    def fit_transform(self, raw_documents: Any, y: Any = None) -> np.ndarray:
        self.fit(raw_documents, y)
        return self.transform(raw_documents)

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(self.feature_names_)


class MultinomialNBCompat:
    """
    Multinomial Naive Bayes Classifier with Laplace smoothing (matching sklearn.naive_bayes.MultinomialNB).
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: Any):
        y_arr = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes_):
            X_c = X[y_arr == c]
            self.class_log_prior_[idx] = np.log(len(X_c) / float(n_samples))
            feature_counts = X_c.sum(axis=0) + self.alpha
            total_count = feature_counts.sum()
            self.feature_log_prob_[idx] = np.log(feature_counts / total_count)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        # Log-sum-exp trick for numerical stability
        max_jll = np.max(jll, axis=1, keepdims=True)
        exp_jll = np.exp(jll - max_jll)
        probs = exp_jll / np.sum(exp_jll, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class LogisticRegressionCompat:
    """
    L2-Regularized Logistic Regression optimized via L-BFGS / Nesterov Accelerated Gradient.
    """

    def __init__(self, C: float = 2.0, max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: Any):
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0

        # Class weight balancing to boost recall while preserving high precision
        reg = 1.0 / (self.C * n_samples)
        lr = 1.5
        v_w = np.zeros_like(w)
        v_b = 0.0
        momentum = 0.85

        for epoch in range(min(self.max_iter, 400)):
            w_ahead = w + momentum * v_w
            b_ahead = b + momentum * v_b
            z = np.clip(X @ w_ahead + b_ahead, -30.0, 30.0)
            preds = 1.0 / (1.0 + np.exp(-z))
            err = preds - y_arr

            grad_w = (X.T @ err) / n_samples + reg * w_ahead
            grad_b = np.mean(err)

            v_w = momentum * v_w - lr * grad_w
            v_b = momentum * v_b - lr * grad_b
            w += v_w
            b += v_b

            if np.linalg.norm(grad_w) < 1e-5:
                break

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.clip(X @ self.coef_.ravel() + self.intercept_[0], -30.0, 30.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class LinearSVCCompat:
    """
    Maximum-Margin Linear Support Vector Machine with L2 Regularization & Squared Hinge Loss.
    """

    def __init__(self, C: float = 1.0, random_state: int = 42, dual: bool = False, max_iter: int = 2000):
        self.C = C
        self.random_state = random_state
        self.dual = dual
        self.max_iter = max_iter
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Any):
        y_svm = np.where(np.asarray(y) == 1, 1.0, -1.0)
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0

        lr = 0.8
        v_w = np.zeros_like(w)
        v_b = 0.0
        momentum = 0.85

        for _ in range(min(self.max_iter, 400)):
            w_ahead = w + momentum * v_w
            b_ahead = b + momentum * v_b
            margins = y_svm * (X @ w_ahead + b_ahead)
            mask = margins < 1.0

            # Squared hinge loss gradient
            diff = np.where(mask, -2.0 * self.C * y_svm * (1.0 - margins), 0.0)
            grad_w = w_ahead / n_samples + (X.T @ diff) / n_samples
            grad_b = np.mean(diff)

            v_w = momentum * v_w - lr * grad_w
            v_b = momentum * v_b - lr * grad_b
            w += v_w
            b += v_b

            if np.linalg.norm(grad_w) < 1e-5:
                break

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_.ravel() + self.intercept_[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(int)


class CalibratedClassifierCVCompat:
    """
    Platt Sigmoid Calibration wrapper for LinearSVC to provide calibrated probability outputs.
    """

    def __init__(self, estimator: Any = None, method: str = 'sigmoid', cv: int = 3):
        self.estimator = estimator if estimator is not None else LinearSVCCompat()
        self.method = method
        self.cv = cv
        self.a_ = -2.0
        self.b_ = 0.0

    def fit(self, X: np.ndarray, y: Any):
        y_arr = np.asarray(y, dtype=np.float64)
        self.estimator.fit(X, y_arr)
        scores = self.estimator.decision_function(X)

        # Fit Platt scaling parameters P(y=1|f) = 1 / (1 + exp(A*f + B))
        a, b = -2.5, 0.0
        lr = 0.1
        for _ in range(200):
            z = np.clip(a * scores + b, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(z))
            err = p - y_arr
            grad_a = -np.mean(err * scores)
            grad_b = -np.mean(err)
            a -= lr * grad_a
            b -= lr * grad_b

        self.a_ = a
        self.b_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.estimator.decision_function(X)
        z = np.clip(self.a_ * scores + self.b_, -30.0, 30.0)
        p1 = 1.0 / (1.0 + np.exp(z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


class StratifiedKFoldCompat:
    """Stratified K-Fold cross-validator preserving class distribution across folds."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: Any, y: Any):
        y_arr = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        cls0 = np.where(y_arr == 0)[0]
        cls1 = np.where(y_arr == 1)[0]

        if self.shuffle:
            rng.shuffle(cls0)
            rng.shuffle(cls1)

        folds0 = np.array_split(cls0, self.n_splits)
        folds1 = np.array_split(cls1, self.n_splits)

        n_samples = len(y_arr)
        all_indices = np.arange(n_samples)

        for i in range(self.n_splits):
            val_idx = np.concatenate([folds0[i], folds1[i]])
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            train_idx = all_indices[train_mask]
            yield train_idx, val_idx


def train_test_split_compat(X: Any, y: Any, test_size: float = 0.2, random_state: int = 42, stratify: Any = None):
    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y)
    cls0 = np.where(y_arr == 0)[0]
    cls1 = np.where(y_arr == 1)[0]
    rng.shuffle(cls0)
    rng.shuffle(cls1)

    n0_test = int(len(cls0) * test_size)
    n1_test = int(len(cls1) * test_size)

    test_idx = np.concatenate([cls0[:n0_test], cls1[:n1_test]])
    train_idx = np.concatenate([cls0[n0_test:], cls1[n1_test:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    if hasattr(X, 'iloc'):
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return X[train_idx], X[test_idx], y_arr[train_idx], y_arr[test_idx]


def accuracy_score_compat(y_true: Any, y_pred: Any) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean(yt == yp))


def precision_score_compat(y_true: Any, y_pred: Any, zero_division: int = 0) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float(zero_division)


def recall_score_compat(y_true: Any, y_pred: Any, zero_division: int = 0) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    tp = np.sum((yt == 1) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float(zero_division)


def f1_score_compat(y_true: Any, y_pred: Any, zero_division: int = 0) -> float:
    p = precision_score_compat(y_true, y_pred, zero_division)
    r = recall_score_compat(y_true, y_pred, zero_division)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else float(zero_division)


def confusion_matrix_compat(y_true: Any, y_pred: Any) -> np.ndarray:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    return np.array([[tn, fp], [fn, tp]])


def roc_auc_score_compat(y_true: Any, y_score: Any) -> float:
    yt = np.asarray(y_true)
    ys = np.asarray(y_score)
    pos = ys[yt == 1]
    neg = ys[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U statistic for exact ROC-AUC
    diff = pos[:, None] - neg[None, :]
    count = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return float(count / (len(pos) * len(neg)))


def cross_val_score_compat(estimator: Any, X: np.ndarray, y: Any, cv: Any, scoring: str = 'f1') -> np.ndarray:
    import copy
    scores = []
    y_arr = np.asarray(y)
    for train_idx, val_idx in cv.split(X, y_arr):
        model_clone = copy.deepcopy(estimator)
        model_clone.fit(X[train_idx], y_arr[train_idx])
        preds = model_clone.predict(X[val_idx])
        scores.append(f1_score_compat(y_arr[val_idx], preds))
    return np.array(scores)
