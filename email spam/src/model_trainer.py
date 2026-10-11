"""
Model Training, Evaluation, Cross-Validation, and Comparison Module.
Implements Naive Bayes, Logistic Regression, and Support Vector Machine (SVM).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )
except ImportError:
    from src.ml_compat import (
        train_test_split_compat as train_test_split,
        StratifiedKFoldCompat as StratifiedKFold,
        cross_val_score_compat as cross_val_score,
        TfidfVectorizerCompat as TfidfVectorizer,
        MultinomialNBCompat as MultinomialNB,
        LogisticRegressionCompat as LogisticRegression,
        LinearSVCCompat as LinearSVC,
        CalibratedClassifierCVCompat as CalibratedClassifierCV,
        accuracy_score_compat as accuracy_score,
        precision_score_compat as precision_score,
        recall_score_compat as recall_score,
        f1_score_compat as f1_score,
        roc_auc_score_compat as roc_auc_score,
        confusion_matrix_compat as confusion_matrix
    )

from src.preprocessing import clean_text
from src.data_loader import load_dataset

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_models() -> Dict[str, Any]:
    """
    Instantiate supervised learning classifiers:
    1. Multinomial Naive Bayes: Classic generative text classifier with Laplace smoothing.
    2. Logistic Regression: Robust discriminative linear model with L2 regularization.
    3. Linear SVM (Calibrated): Maximum-margin hyperplane classifier with sigmoid probability calibration.
    """
    return {
        "Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42, dual=False, max_iter=2000),
            method='sigmoid',
            cv=3
        )
    }


def train_and_evaluate_all() -> Dict[str, Any]:
    """
    Executes end-to-end training and evaluation:
    1. Loads dataset
    2. Preprocesses messages
    3. Splits data into train and test sets BEFORE vectorization (Strict Leakage Prevention)
    4. Fits TF-IDF on train set only, transforms test set
    5. Trains and cross-validates each model
    6. Computes comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    7. Identifies top spam and ham predictive words
    8. Serializes artifacts for production inference
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_dataset()

    print("[*] Preprocessing text corpus (cleaning, stemming, tokenization)...")
    df['cleaned_message'] = df['message'].apply(clean_text)

    # Filter out empty messages if any remain after cleaning
    df = df[df['cleaned_message'].str.len() > 0].copy()

    X = df['cleaned_message']
    y = df['target']

    # Step 3: Strict split BEFORE featurization to prevent data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"[*] Training samples: {len(X_train_raw)}, Testing samples: {len(X_test_raw)}")

    # Step 4: Fit TF-IDF Vectorizer solely on training set
    print("[*] Vectorizing text with TF-IDF (unigrams + bigrams, max_features=5000)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)

    models = build_models()
    results = {}
    fitted_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n==================== Training {name} ====================")
        
        # 5-fold cross validation on training data
        cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='f1')
        print(f"[*] 5-Fold Cross-Validation F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Fit model on full training set
        model.fit(X_train_vec, y_train)
        fitted_models[name] = model

        # Predict on hold-out test set
        y_pred = model.predict(X_test_vec)
        y_prob = model.predict_proba(X_test_vec)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)

        # In spam filtering, False Positives (FP) are legitimate emails wrongfully classified as spam
        tn, fp, fn, tp = cm.ravel()

        print(f"    - Accuracy:         {acc * 100:.2f}%")
        print(f"    - Precision (Spam): {prec * 100:.2f}%")
        print(f"    - Recall (Spam):    {rec * 100:.2f}%")
        print(f"    - F1-Score:         {f1 * 100:.2f}%")
        print(f"    - ROC-AUC:          {roc:.4f}")
        print(f"    - Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }

    # Model Selection: Prioritize high precision (low FP) followed by F1-Score
    best_model_name = max(results.keys(), key=lambda k: (results[k]['precision'], results[k]['f1_score']))
    print(f"\n[+] Selected Optimal Model: {best_model_name}")

    # Extract Top Predictive Features from MultinomialNB & Logistic Regression
    feature_names = np.array(vectorizer.get_feature_names_out())
    nb_model = fitted_models["Naive Bayes"]
    lr_model = fitted_models["Logistic Regression"]

    # Log ratio for Naive Bayes
    log_ratio = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]
    top_spam_indices_nb = np.argsort(log_ratio)[-25:][::-1]
    top_ham_indices_nb = np.argsort(log_ratio)[:25]

    top_spam_keywords = feature_names[top_spam_indices_nb].tolist()
    top_ham_keywords = feature_names[top_ham_indices_nb].tolist()

    # Save artifacts
    print("[*] Serializing vectorizer and model artifacts...")
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    for name, model in fitted_models.items():
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{slug}_model.joblib"))

    # Save best model reference
    joblib.dump(fitted_models[best_model_name], os.path.join(MODELS_DIR, "best_model.joblib"))

    summary_data = {
        "best_model": best_model_name,
        "models": results,
        "top_spam_keywords": top_spam_keywords,
        "top_ham_keywords": top_ham_keywords,
        "dataset_stats": {
            "total_records": len(df),
            "train_records": len(X_train_raw),
            "test_records": len(X_test_raw),
            "spam_count": int(df['target'].sum()),
            "ham_count": int(len(df) - df['target'].sum())
        }
    }

    metrics_file = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print(f"[+] Evaluation summary saved to {metrics_file}")
    return summary_data
