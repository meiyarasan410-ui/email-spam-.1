"""
Flask Web Application for Email Spam Classifier.
Provides an interactive real-time user interface, live prediction API,
visual dashboard, and preloaded sample testing.
"""

import os
import json
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from src.classifier import SpamClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)

# Cache classifier instance
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        try:
            classifier = SpamClassifier()
        except Exception:
            from src.model_trainer import train_and_evaluate_all
            from src.visualizer import generate_all_plots
            train_and_evaluate_all()
            generate_all_plots()
            classifier = SpamClassifier()
    return classifier


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Spam Classifier | Machine Learning & NLP</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-dark: #4338ca;
            --spam-color: #ef4444;
            --ham-color: #10b981;
            --bg-color: #f8fafc;
        }
        body {
            background-color: var(--bg-color);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            color: #1e293b;
        }
        .navbar-brand {
            font-weight: 700;
            color: var(--primary) !important;
            letter-spacing: -0.5px;
        }
        .card {
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            background: #ffffff;
        }
        .badge-spam {
            background-color: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fca5a5;
            font-size: 1.1rem;
            padding: 8px 16px;
        }
        .badge-ham {
            background-color: #d1fae5;
            color: #047857;
            border: 1px solid #6ee7b7;
            font-size: 1.1rem;
            padding: 8px 16px;
        }
        .trigger-tag {
            display: inline-block;
            background: #fef3c7;
            color: #92400e;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 2px;
            border: 1px solid #fde68a;
        }
        .progress {
            height: 12px;
            border-radius: 6px;
        }
        .sample-btn {
            font-size: 0.85rem;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .nav-tabs .nav-link.active {
            font-weight: 600;
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
        }
        .plot-img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
    </style>
</head>
<body>

<nav class="navbar navbar-expand-lg navbar-light bg-white border-bottom sticky-top">
    <div class="container">
        <a class="navbar-brand" href="#">
            <i class="fa-solid fa-shield-halved text-primary me-2"></i>Email Spam Classifier
        </a>
        <span class="badge bg-light text-dark border">Naive Bayes • Logistic Regression • SVM</span>
    </div>
</nav>

<div class="container py-4">
    <ul class="nav nav-tabs mb-4" id="mainTab" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="classifier-tab" data-bs-toggle="tab" data-bs-target="#classifier-pane" type="button" role="tab">
                <i class="fa-solid fa-envelope me-1"></i> Real-Time Classifier
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="metrics-tab" data-bs-toggle="tab" data-bs-target="#metrics-pane" type="button" role="tab">
                <i class="fa-solid fa-chart-column me-1"></i> Model Benchmarks & Metrics
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="visuals-tab" data-bs-toggle="tab" data-bs-target="#visuals-pane" type="button" role="tab">
                <i class="fa-solid fa-image me-1"></i> Keyword & EDA Visualizations
            </button>
        </li>
    </ul>

    <div class="tab-content" id="mainTabContent">
        <!-- PANE 1: REAL-TIME CLASSIFIER -->
        <div class="tab-pane fade show active" id="classifier-pane" role="tabpanel">
            <div class="row g-4">
                <div class="col-lg-7">
                    <div class="card p-4 h-100">
                        <h5 class="fw-bold mb-3"><i class="fa-solid fa-paste me-2 text-primary"></i>Email / SMS Content Input</h5>
                        <p class="text-muted small">Enter or paste message content below to classify in real-time using NLP feature extraction and calibrated ML inference.</p>
                        
                        <div class="mb-3">
                            <label class="form-label fw-semibold">Quick Pre-loaded Samples:</label>
                            <div>
                                <button class="btn btn-sm btn-outline-danger sample-btn" onclick="loadSample('spam1')">
                                    <i class="fa-solid fa-triangle-exclamation"></i> Lottery Winner Spam
                                </button>
                                <button class="btn btn-sm btn-outline-danger sample-btn" onclick="loadSample('spam2')">
                                    <i class="fa-solid fa-credit-card"></i> Bank Alert Phishing
                                </button>
                                <button class="btn btn-sm btn-outline-success sample-btn" onclick="loadSample('ham1')">
                                    <i class="fa-solid fa-briefcase"></i> Work Meeting Ham
                                </button>
                                <button class="btn btn-sm btn-outline-success sample-btn" onclick="loadSample('ham2')">
                                    <i class="fa-solid fa-calendar"></i> Lunch Invite Ham
                                </button>
                            </div>
                        </div>

                        <div class="mb-3">
                            <textarea id="emailInput" class="form-control" rows="8" placeholder="Paste incoming email header, subject, or message body here..."></textarea>
                        </div>

                        <div class="d-flex justify-content-between align-items-center">
                            <button id="classifyBtn" class="btn btn-primary px-4" onclick="classifyEmail()">
                                <i class="fa-solid fa-wand-magic-sparkles me-1"></i> Classify Email
                            </button>
                            <button class="btn btn-outline-secondary btn-sm" onclick="clearInput()">Clear</button>
                        </div>
                    </div>
                </div>

                <div class="col-lg-5">
                    <div class="card p-4 h-100" id="resultCard">
                        <h5 class="fw-bold mb-3"><i class="fa-solid fa-square-poll-vertical me-2 text-primary"></i>Classification Assessment</h5>
                        
                        <div id="emptyResult" class="text-center py-5 text-muted">
                            <i class="fa-solid fa-inbox fa-3x mb-3 text-secondary opacity-50"></i>
                            <p>No email analyzed yet.<br>Click "Classify Email" or select a sample above.</p>
                        </div>

                        <div id="activeResult" style="display: none;">
                            <div class="text-center mb-4">
                                <span id="labelBadge" class="badge rounded-pill fw-bold"></span>
                                <h6 class="text-muted mt-2" id="riskLevelText"></h6>
                            </div>

                            <div class="mb-3">
                                <div class="d-flex justify-content-between small fw-semibold mb-1">
                                    <span>Spam Likelihood</span>
                                    <span id="probPercent">0%</span>
                                </div>
                                <div class="progress">
                                    <div id="probBar" class="progress-bar" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <div class="d-flex justify-content-between small fw-semibold mb-1">
                                    <span>Model Confidence</span>
                                    <span id="confPercent">0%</span>
                                </div>
                                <div class="progress">
                                    <div id="confBar" class="progress-bar bg-info" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="fw-semibold small text-muted d-block mb-1">Detected Suspicious Keywords / Triggers:</label>
                                <div id="triggerContainer"></div>
                            </div>

                            <div class="border-top pt-3 mt-3">
                                <label class="fw-semibold small text-muted d-block mb-1">Structural Meta Features:</label>
                                <div class="row g-2 small text-muted" id="metaContainer">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- PANE 2: MODEL BENCHMARKS -->
        <div class="tab-pane fade" id="metrics-pane" role="tabpanel">
            <div class="card p-4">
                <h5 class="fw-bold mb-3"><i class="fa-solid fa-trophy text-warning me-2"></i>Algorithms Evaluated & Compared</h5>
                <p class="text-muted">Rigorous multi-model comparison across Naive Bayes, Logistic Regression, and Support Vector Machine on holdout test partition.</p>
                
                <div class="table-responsive">
                    <table class="table table-bordered table-hover align-middle">
                        <thead class="table-light">
                            <tr>
                                <th>Algorithm</th>
                                <th>Accuracy</th>
                                <th>Precision (Spam)</th>
                                <th>Recall (Spam)</th>
                                <th>F1-Score</th>
                                <th>ROC-AUC</th>
                                <th>False Positives (FP)</th>
                            </tr>
                        </thead>
                        <tbody id="metricsTableBody">
                            <tr><td colspan="7" class="text-center">Loading model benchmarks...</td></tr>
                        </tbody>
                    </table>
                </div>

                <div class="alert alert-info mt-3">
                    <i class="fa-solid fa-circle-info me-2"></i>
                    <strong>False Positive Priority:</strong> In spam classification, minimizing False Positives (legitimate emails marked as spam) is paramount to preventing loss of critical personal or business communications.
                </div>
            </div>
        </div>

        <!-- PANE 3: VISUALIZATIONS -->
        <div class="tab-pane fade" id="visuals-pane" role="tabpanel">
            <div class="row g-4">
                <div class="col-md-6">
                    <div class="card p-3">
                        <h6 class="fw-bold mb-2">Model Comparison Summary</h6>
                        <img src="/plots/model_comparison.png" class="plot-img" onerror="this.src='https://via.placeholder.com/600x400?text=Plot+Generating...'">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card p-3">
                        <h6 class="fw-bold mb-2">Confusion Matrix Heatmaps</h6>
                        <img src="/plots/confusion_matrices.png" class="plot-img" onerror="this.src='https://via.placeholder.com/600x400?text=Plot+Generating...'">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card p-3">
                        <h6 class="fw-bold mb-2">Top Discriminative Keywords</h6>
                        <img src="/plots/keyword_importance.png" class="plot-img" onerror="this.src='https://via.placeholder.com/600x400?text=Plot+Generating...'">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card p-3">
                        <h6 class="fw-bold mb-2">Message Length Distributions</h6>
                        <img src="/plots/message_distributions.png" class="plot-img" onerror="this.src='https://via.placeholder.com/600x400?text=Plot+Generating...'">
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
    const SAMPLES = {
        'spam1': 'CONGRATULATIONS! You have been selected as the guaranteed winner of a £1,000 cash prize or Walmart gift card! Call 09061701461 now or click http://claim-prize.net to claim your award immediately. Limited time only!',
        'spam2': 'URGENT SECURITY ALERT: Your bank account has been suspended due to suspicious activity. Verify your login password and identity immediately at http://secure-ebank-update.com/login or access will be terminated.',
        'ham1': 'Hi Team, please find attached the weekly engineering progress report. Let us review the key deliverables during tomorrow morning sync at 10 AM in Conference Room B. Thanks, Alex.',
        'ham2': 'Hey, are we still meeting for lunch today at 12:30 PM near the cafeteria? Let me know if that time works for you.'
    };

    function loadSample(key) {
        document.getElementById('emailInput').value = SAMPLES[key] || '';
        classifyEmail();
    }

    function clearInput() {
        document.getElementById('emailInput').value = '';
        document.getElementById('emptyResult').style.display = 'block';
        document.getElementById('activeResult').style.display = 'none';
    }

    async function classifyEmail() {
        const text = document.getElementById('emailInput').value.trim();
        if (!text) {
            alert('Please enter or paste an email message to classify.');
            return;
        }

        const btn = document.getElementById('classifyBtn');
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Analyzing...';
        btn.disabled = true;

        try {
            const resp = await fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await resp.json();

            document.getElementById('emptyResult').style.display = 'none';
            document.getElementById('activeResult').style.display = 'block';

            const badge = document.getElementById('labelBadge');
            const probBar = document.getElementById('probBar');
            const confBar = document.getElementById('confBar');
            
            if (data.is_spam) {
                badge.className = 'badge badge-spam';
                badge.innerHTML = '<i class="fa-solid fa-triangle-exclamation me-2"></i>SPAM DETECTED';
                probBar.className = 'progress-bar bg-danger';
            } else {
                badge.className = 'badge badge-ham';
                badge.innerHTML = '<i class="fa-solid fa-circle-check me-2"></i>HAM (LEGITIMATE)';
                probBar.className = 'progress-bar bg-success';
            }

            document.getElementById('riskLevelText').innerText = 'Risk Level: ' + data.risk_level;
            
            const spamProbPct = (data.spam_probability * 100).toFixed(1) + '%';
            document.getElementById('probPercent').innerText = spamProbPct;
            probBar.style.width = spamProbPct;

            const confPct = (data.confidence * 100).toFixed(1) + '%';
            document.getElementById('confPercent').innerText = confPct;
            confBar.style.width = confPct;

            // Trigger words
            const trigCont = document.getElementById('triggerContainer');
            trigCont.innerHTML = '';
            if (data.detected_triggers && data.detected_triggers.length > 0) {
                data.detected_triggers.forEach(word => {
                    const span = document.createElement('span');
                    span.className = 'trigger-tag';
                    span.innerText = word;
                    trigCont.appendChild(span);
                });
            } else {
                trigCont.innerHTML = '<span class="text-muted small fst-italic">No known trigger keywords detected</span>';
            }

            // Meta features
            const metaCont = document.getElementById('metaContainer');
            metaCont.innerHTML = `
                <div class="col-6">Characters: <strong>${data.meta_features.char_count}</strong></div>
                <div class="col-6">Words: <strong>${data.meta_features.word_count}</strong></div>
                <div class="col-6">Uppercase Ratio: <strong>${(data.meta_features.uppercase_ratio * 100).toFixed(1)}%</strong></div>
                <div class="col-6">Exclamation Marks: <strong>${data.meta_features.exclamation_count}</strong></div>
                <div class="col-6">Currency Symbols: <strong>${data.meta_features.currency_count}</strong></div>
                <div class="col-6">Contains URL: <strong>${data.meta_features.has_url ? 'Yes' : 'No'}</strong></div>
            `;

        } catch (err) {
            console.error(err);
            alert('Failed to connect to classification engine.');
        } finally {
            btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles me-1"></i> Classify Email';
            btn.disabled = false;
        }
    }

    async function fetchMetrics() {
        try {
            const resp = await fetch('/api/metrics');
            const data = await resp.json();
            const tbody = document.getElementById('metricsTableBody');
            tbody.innerHTML = '';

            for (const [modelName, m] of Object.entries(data.models)) {
                const tr = document.createElement('tr');
                const isBest = (modelName === data.best_model);
                tr.innerHTML = `
                    <td><strong>${modelName}</strong> ${isBest ? '<span class="badge bg-success ms-1">Recommended</span>' : ''}</td>
                    <td>${(m.accuracy * 100).toFixed(2)}%</td>
                    <td><strong>${(m.precision * 100).toFixed(2)}%</strong></td>
                    <td>${(m.recall * 100).toFixed(2)}%</td>
                    <td>${(m.f1_score * 100).toFixed(2)}%</td>
                    <td>${(m.roc_auc).toFixed(4)}</td>
                    <td><span class="badge ${m.confusion_matrix.fp === 0 ? 'bg-success' : 'bg-warning'}">${m.confusion_matrix.fp}</span></td>
                `;
                tbody.appendChild(tr);
            }
        } catch (e) {
            console.error('Metrics fetch error:', e);
        }
    }

    window.addEventListener('DOMContentLoaded', () => {
        fetchMetrics();
    });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    clf = get_classifier()
    result = clf.predict(text)
    return jsonify(result)


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    metrics_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Metrics not found. Please train models first."}), 404


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)


if __name__ == "__main__":
    get_classifier()
    app.run(host="127.0.0.1", port=5000, debug=True)
