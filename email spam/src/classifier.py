"""
Inference Engine and Real-Time Email Spam Classifier.
Provides single-message and batch classification with risk scoring and keyword highlighting.
"""

import os
import joblib
from typing import Dict, Any, List, Optional
from src.preprocessing import clean_text, extract_meta_features, get_spam_trigger_words

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class SpamClassifier:
    """
    Lightweight, fast inference engine for real-time email spam detection.
    Can be seamlessly embedded into mail agents, APIs, or interactive UIs.
    """

    def __init__(self, model_name: str = "best_model"):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized ML model and TF-IDF vectorizer."""
        vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
        if not os.path.exists(vec_path):
            raise FileNotFoundError(
                f"Vectorizer not found at {vec_path}. Please train models first using 'python main.py --train'."
            )

        model_file = f"{self.model_name}.joblib" if not self.model_name.endswith(".joblib") else self.model_name
        model_path = os.path.join(MODELS_DIR, model_file)
        if not os.path.exists(model_path):
            # Fallback to best_model if custom slug not found
            model_path = os.path.join(MODELS_DIR, "best_model.joblib")

        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies an email or message in real time.
        
        Args:
            text: Raw message content (subject + body).
            
        Returns:
            Dictionary with classification label, probability, risk assessment,
            trigger words, and extracted structural metadata.
        """
        if not text or not text.strip():
            return {
                "label": "HAM",
                "is_spam": False,
                "spam_probability": 0.0,
                "confidence": 1.0,
                "risk_level": "None",
                "detected_triggers": [],
                "meta_features": extract_meta_features(""),
                "cleaned_text": ""
            }

        meta = extract_meta_features(text)
        cleaned = clean_text(text)
        
        # Transform using pre-fitted TF-IDF
        vec = self.vectorizer.transform([cleaned])
        
        # Predict class
        pred = self.model.predict(vec)[0]
        
        # Predict probability if supported
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            spam_prob = float(probs[1])
        elif hasattr(self.model, "decision_function"):
            df_val = float(self.model.decision_function(vec)[0])
            # Sigmoid approximation
            import math
            spam_prob = 1.0 / (1.0 + math.exp(-df_val))
        else:
            spam_prob = 1.0 if pred == 1 else 0.0

        is_spam = bool(pred == 1)
        confidence = spam_prob if is_spam else (1.0 - spam_prob)

        # Detect spam trigger words from the message
        all_triggers = get_spam_trigger_words()
        text_lower = text.lower()
        found_triggers = [w for w in all_triggers if w in text_lower or (w in cleaned.split())]

        # Determine risk tier
        if spam_prob >= 0.85:
            risk_level = "Critical Spam / Phishing"
        elif spam_prob >= 0.60:
            risk_level = "High Spam Risk"
        elif spam_prob >= 0.35:
            risk_level = "Suspicious / Moderate"
        else:
            risk_level = "Clean / Legitimate"

        return {
            "label": "SPAM" if is_spam else "HAM",
            "is_spam": is_spam,
            "spam_probability": round(spam_prob, 4),
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "detected_triggers": sorted(list(set(found_triggers))),
            "meta_features": meta,
            "cleaned_text": cleaned
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch predict messages for bulk email system integration."""
        return [self.predict(t) for t in texts]
