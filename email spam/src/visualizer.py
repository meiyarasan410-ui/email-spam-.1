"""
Visualization Module for Email Spam Classifier.
Generates publication-quality charts for model comparison, keyword importance,
confusion matrices, and exploratory data distributions using Matplotlib and Seaborn.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.data_loader import load_dataset

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "plots")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def generate_all_plots(metrics_path: str = None) -> None:
    """
    Generate and save all diagnostic and comparison visual charts.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if metrics_path is None:
        metrics_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")

    if not os.path.exists(metrics_path):
        print(f"[!] Warning: Metrics file {metrics_path} does not exist. Cannot generate model plots.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    sns.set_theme(style="whitegrid")
    
    _plot_model_comparison(metrics)
    _plot_confusion_matrices(metrics)
    _plot_top_keywords(metrics)
    _plot_length_distributions()
    print(f"[+] All visual plots generated successfully in {PLOTS_DIR}")


def _plot_model_comparison(metrics: Dict[str, Any]) -> None:
    """Bar chart comparing Accuracy, Precision, Recall, and F1 across models."""
    models_data = metrics.get("models", {})
    records = []
    for model_name, scores in models_data.items():
        records.append({"Model": model_name, "Metric": "Accuracy", "Score": scores["accuracy"]})
        records.append({"Model": model_name, "Metric": "Precision (Spam)", "Score": scores["precision"]})
        records.append({"Model": model_name, "Metric": "Recall (Spam)", "Score": scores["recall"]})
        records.append({"Model": model_name, "Metric": "F1-Score", "Score": scores["f1_score"]})

    df_plot = pd.DataFrame(records)

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("deep", 4)
    ax = sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette=palette)
    plt.title("Algorithm Performance Comparison (SMS / Email Spam Classification)", fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0.75, 1.02)
    plt.ylabel("Performance Score (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Supervised Learning Algorithm", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()

    # Annotate values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 3),
                        textcoords='offset points')

    out_file = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"    - Saved: {out_file}")


def _plot_confusion_matrices(metrics: Dict[str, Any]) -> None:
    """Heatmaps of Confusion Matrices for each algorithm."""
    models_data = metrics.get("models", {})
    num_models = len(models_data)
    if num_models == 0:
        return

    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4.5))
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, scores) in zip(axes, models_data.items()):
        cm_dict = scores["confusion_matrix"]
        cm_matrix = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]]
        ])

        sns.heatmap(
            cm_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Ham (0)", "Spam (1)"],
            yticklabels=["Ham (0)", "Spam (1)"]
        )
        ax.set_title(f"{model_name}\nPrecision: {scores['precision']*100:.1f}% | FP: {cm_dict['fp']}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.suptitle("Confusion Matrix Analysis on Test Set (Emphasizing Low False Positives)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_file = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - Saved: {out_file}")


def _plot_top_keywords(metrics: Dict[str, Any]) -> None:
    """Horizontal bar chart showing top distinctive spam vs. ham keywords."""
    top_spam = metrics.get("top_spam_keywords", [])[:15]
    top_ham = metrics.get("top_ham_keywords", [])[:15]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Reverse for top-down display
    ax1.barh(range(len(top_spam)), [1] * len(top_spam), color="#d9534f", alpha=0.85)
    ax1.set_yticks(range(len(top_spam)))
    ax1.set_yticklabels(top_spam, fontsize=11, fontweight='semibold')
    ax1.set_title("Top Spam Discriminative Keywords", fontsize=12, fontweight='bold', color="#a94442")
    ax1.invert_yaxis()
    ax1.get_xaxis().set_visible(False)

    ax2.barh(range(len(top_ham)), [1] * len(top_ham), color="#5cb85c", alpha=0.85)
    ax2.set_yticks(range(len(top_ham)))
    ax2.set_yticklabels(top_ham, fontsize=11, fontweight='semibold')
    ax2.set_title("Top Legitimate (Ham) Keywords", fontsize=12, fontweight='bold', color="#3c763d")
    ax2.invert_yaxis()
    ax2.get_xaxis().set_visible(False)

    plt.suptitle("Discriminative Keyword Signatures Identified by Naive Bayes", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_file = os.path.join(PLOTS_DIR, "keyword_importance.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - Saved: {out_file}")


def _plot_length_distributions() -> None:
    """Plots message character and word length distribution comparing Spam vs Ham."""
    try:
        df = load_dataset()
        df['char_length'] = df['message'].apply(len)
        df['word_count'] = df['message'].apply(lambda s: len(s.split()))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(
            data=df, x="char_length", hue="label",
            bins=50, kde=True, element="step",
            palette={"ham": "#337ab7", "spam": "#d9534f"},
            ax=ax1
        )
        ax1.set_title("Character Length Distribution (Spam vs. Ham)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Character Count")
        ax1.set_xlim(0, 350)

        sns.boxplot(
            data=df, x="label", y="word_count",
            palette={"ham": "#337ab7", "spam": "#d9534f"},
            ax=ax2
        )
        ax2.set_title("Word Count Boxplot", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Word Count")
        ax2.set_ylim(0, 60)

        plt.suptitle("Exploratory Analysis: Structural Characteristics of Messages", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_file = os.path.join(PLOTS_DIR, "message_distributions.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    - Saved: {out_file}")
    except Exception as e:
        print(f"[!] Warning: Could not plot message distribution: {e}")
