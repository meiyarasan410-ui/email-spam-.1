# 📧 Email & SMS Spam Classifier

An end-to-end, production-grade Machine Learning and Natural Language Processing (NLP) system designed to accurately identify and filter spam, promotional, and malicious communications with an emphasis on **ultra-low false-positive rates**.

---

## 📌 Project Overview

In the modern digital communication landscape, spam emails flood inboxes, consume bandwidth, and expose users to phishing attacks, scams, and financial fraud. 

This project implements a complete, modular machine learning pipeline capable of:
1. **Cleaning and normalizing raw email text** (stripping markup, standardizing currency, URLs, phone numbers, and performing morphological stemming).
2. **Feature extraction** via Term Frequency-Inverse Document Frequency (TF-IDF) unigrams and bigrams.
3. **Training and comparing three supervised learning algorithms**:
   - **Multinomial Naive Bayes (MNB)**
   - **Logistic Regression (LR)**
   - **Support Vector Machine (Linear SVM)** with calibrated probability estimation
4. **Minimizing False Positives**: In spam classification, misclassifying a legitimate (ham) email as spam can lead to catastrophic business or personal communication loss. Our models and thresholds are engineered to prioritize spam precision.
5. **Interactive Real-Time Inference**: Provided via an interactive command-line shell, one-shot CLI predictions, and a responsive Flask web application.

---

## 🗂️ Project Architecture

```
email-spam-classifier/
├── data/                       # Dataset storage (SMS Spam Collection / spam.csv)
│   └── spam.csv
├── models/                     # Serialized model weights & vectorizers
│   ├── tfidf_vectorizer.joblib
│   ├── best_model.joblib
│   ├── naive_bayes_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── support_vector_machine_model.joblib
│   └── evaluation_metrics.json
├── outputs/
│   └── plots/                  # Generated diagnostic visual charts
│       ├── model_comparison.png
│       ├── confusion_matrices.png
│       ├── keyword_importance.png
│       └── message_distributions.png
├── src/                        # Core algorithmic modules
│   ├── __init__.py
│   ├── data_loader.py          # Automated dataset ingestion and schema parsing
│   ├── preprocessing.py       # Regex normalization, stopword filtering, stemming
│   ├── model_trainer.py        # Leakage-free training, StratifiedKFold CV, model export
│   ├── classifier.py           # Real-time inference engine and risk scoring
│   └── visualizer.py           # Matplotlib & Seaborn chart generators
├── tests/                      # Test suite
│   └── test_pipeline.py
├── app.py                      # Interactive Flask web application & REST API
├── main.py                     # Unified CLI entrypoint
├── requirements.txt            # Locked project dependencies
└── README.md                   # System documentation
```

---

## ⚙️ Mathematical Formulations & Algorithms

### 1. Multinomial Naive Bayes
Applies Bayes' Theorem under the conditional independence assumption:
$$P(y \mid \mathbf{x}) = \frac{P(y) \prod_{i=1}^n P(x_i \mid y)^{x_i}}{P(\mathbf{x})}$$
With Laplace smoothing ($\alpha$):
$$\hat{\theta}_{yi} = \frac{N_{yi} + \alpha}{N_y + \alpha n}$$
- **Strengths**: Extremely fast, strong baseline for high-dimensional sparse text vectors, resistant to overfitting on small corpora.

### 2. Logistic Regression
Models the posterior log-odds of a message being spam as a linear combination of feature weights:
$$\log \left( \frac{P(y=1 \mid \mathbf{x})}{1 - P(y=1 \mid \mathbf{x})} \right) = \mathbf{w}^T \mathbf{x} + b$$
$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Strengths**: Well-calibrated probabilities, interpretable feature coefficients, highly effective with $L_2$ regularization.

### 3. Support Vector Machine (Linear SVM)
Solves for the maximum-margin hyperplane separating spam from legitimate messages:
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^m \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$
- **Probability Calibration**: Calibrated using Platt's sigmoid scaling via cross-validation to provide probabilistic confidence scores.
- **Strengths**: Excels in high-dimensional TF-IDF feature spaces where linear separability is pronounced.

---

## 🚀 Getting Started

### 1. Activate Environment & Install Dependencies

From the project root:
```bash
# Windows
.venv\Scripts\activate

# Or install dependencies if setting up afresh
pip install -r requirements.txt
```

### 2. Train Models and Generate Visualizations

To train all three models with 5-fold cross-validation and generate all diagnostic plots:
```bash
python main.py --train
```

### 3. Real-Time Single Message Prediction

Classify any message string directly from your terminal:
```bash
python main.py --predict "CONGRATULATIONS! You won a $1,000 Walmart Gift Card! Click here to claim."
```

### 4. Interactive Terminal Shell

Launch an interactive prompt to test custom messages continuously:
```bash
python main.py --interactive
```

### 5. Launch the Web Application Dashboard

Launch the Flask web interface:
```bash
python main.py --serve --port 5000
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to:
- Test emails with live confidence bars and risk indicators.
- Inspect detected trigger keywords.
- Review comparative model benchmark tables.
- View interactive diagnostic plots (confusion matrices, keyword importance).

---

## 🧪 Running Unit Tests

Run the test suite to verify preprocessing normalization, feature extraction, and dataset integrity:
```bash
python -m unittest discover tests
```

---

## 🔮 Future Extensions
- **Phishing & Brand Impersonation Detection**: Integrating domain-reputation lookups and Typosquatting checks.
- **Header Analysis**: Parsing SPF, DKIM, and DMARC verification records from raw `.eml` headers.
- **Transformer Fine-Tuning**: Fine-tuning lightweight transformer backbones (DistilBERT or RoBERTa) for contextual nuance.
