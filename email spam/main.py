"""
Main CLI Application & Pipeline Orchestrator for Email Spam Classifier.
"""

import sys
import os
import argparse
import json

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_dataset
from src.model_trainer import train_and_evaluate_all
from src.visualizer import generate_all_plots
from src.classifier import SpamClassifier


def print_banner():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    banner = """
+==================================================================+
|               EMAIL & SMS SPAM CLASSIFIER SYSTEM                 |
|       Supervised ML with Naive Bayes, Logistic Reg, & SVM        |
+==================================================================+
    """
    print(banner)


def run_training_pipeline():
    print("[*] Initiating complete training & evaluation pipeline...")
    metrics = train_and_evaluate_all()
    print("[*] Generating visualization plots...")
    generate_all_plots()
    print("\n[+] Pipeline execution finished successfully!")
    print(f"    - Best Model Selected: {metrics['best_model']}")
    for model_name, res in metrics['models'].items():
        print(f"    - {model_name:24s} | Acc: {res['accuracy']*100:.2f}% | Prec: {res['precision']*100:.2f}% | F1: {res['f1_score']*100:.2f}% | FP: {res['confusion_matrix']['fp']}")


def run_prediction(text: str):
    try:
        classifier = SpamClassifier()
        res = classifier.predict(text)
        
        status_color = "[SPAM - MALICIOUS/PROMOTIONAL]" if res['is_spam'] else "[HAM - LEGITIMATE]"
        print("\n" + "="*60)
        print(f"CLASSIFICATION RESULT: {status_color}")
        print("="*60)
        print(f"* Input Text:        \"{text}\"")
        print(f"* Predicted Label:   {res['label']}")
        print(f"* Spam Probability:  {res['spam_probability']*100:.2f}%")
        print(f"* Confidence:        {res['confidence']*100:.2f}%")
        print(f"* Assessed Risk:     {res['risk_level']}")
        if res['detected_triggers']:
            print(f"* Trigger Keywords:  {', '.join(res['detected_triggers'])}")
        else:
            print(f"* Trigger Keywords:  None detected")
        print(f"* Preprocessed Text: \"{res['cleaned_text']}\"")
        print(f"* Message Length:    {res['meta_features']['char_count']} chars, {res['meta_features']['word_count']} words")
        print("="*60 + "\n")
    except Exception as e:
        print(f"[!] Error during prediction: {e}")
        print("    Tip: Run 'python main.py --train' first to train and build models.")


def run_interactive_mode():
    try:
        classifier = SpamClassifier()
    except Exception as e:
        print(f"[!] Model not found. Training model first...")
        run_training_pipeline()
        classifier = SpamClassifier()

    print("\n[+] Entering Interactive Spam Classification Shell.")
    print("    Type your email or message below. Type 'quit' or 'exit' to return.\n")

    while True:
        try:
            user_input = input("Enter email text > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("[*] Exiting interactive shell. Goodbye!")
                break
            run_prediction(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\n[*] Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Machine Learning Email Spam Classifier CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", action="store_true", help="Execute end-to-end data processing, model training, and plot generation")
    parser.add_argument("--predict", type=str, help="Classify a single email or message string")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal prediction mode")
    parser.add_argument("--visualize", action="store_true", help="Generate or regenerate diagnostic Matplotlib/Seaborn plots")
    parser.add_argument("--serve", action="store_true", help="Launch the interactive Flask Web Application dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port to run web app (default: 5000)")

    args = parser.parse_args()

    print_banner()

    if args.train:
        run_training_pipeline()
    elif args.visualize:
        generate_all_plots()
    elif args.predict:
        run_prediction(args.predict)
    elif args.interactive:
        run_interactive_mode()
    elif args.serve:
        from app import app
        print(f"[*] Starting Email Spam Classifier Web Dashboard on http://127.0.0.1:{args.port} ...")
        app.run(host="127.0.0.1", port=args.port, debug=False)
    else:
        # Default behavior if no arguments given: show help and quick interactive prompt
        parser.print_help()


if __name__ == "__main__":
    main()
