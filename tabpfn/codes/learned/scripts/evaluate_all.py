"""Stage 5: Evaluate all methods on OpenML datasets.

Compares:
1. Broad base PFN
2. Individual specialists
3. Soft evidence-weighted portfolio
4. Hard selection (argmax specialist)
5. Linear support-adaptive merge
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import accuracy_score, log_loss

from tfmplayground.evaluation import get_openml_predictions, TOY_TASKS_CLASSIFICATION
from tfmplayground.interface import NanoTabPFNClassifier

from src.utils import load_model_from_checkpoint, load_specialists
from src.evidence import compute_specialist_weights
from src.merging import linear_merge


def evaluate_model(model, device, tasks, label=""):
    """Evaluate a single model on tasks and print results."""
    classifier = NanoTabPFNClassifier(model, device)
    predictions = get_openml_predictions(model=classifier, tasks=tasks)

    accs = []
    for name, (y_true, y_pred, y_proba) in predictions.items():
        acc = accuracy_score(y_true, y_pred)
        accs.append(acc)
    avg_acc = np.mean(accs)
    print(f"  {label}: avg accuracy = {avg_acc:.4f} ({len(accs)} tasks)")
    return avg_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate all methods")
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--specialist_scm", type=str, required=True)
    parser.add_argument("--specialist_tree", type=str, required=True)
    parser.add_argument("--specialist_nn", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specialist_paths = [args.specialist_scm, args.specialist_tree, args.specialist_nn]
    family_names = ["SCM", "Tree", "NN"]

    base_model, specialist_models, base_state, specialist_states = load_specialists(
        args.base_checkpoint, specialist_paths, device
    )

    tasks = TOY_TASKS_CLASSIFICATION
    print("=" * 60)
    print("Evaluation on TOY_TASKS_CLASSIFICATION")
    print("=" * 60)

    # 1. Broad base
    print("\n[1] Broad base PFN:")
    evaluate_model(base_model, device, tasks, "Base")

    # 2. Individual specialists
    print("\n[2] Individual specialists:")
    for name, model in zip(family_names, specialist_models):
        evaluate_model(model, device, tasks, f"Specialist-{name}")

    # 3-5. Methods requiring support evidence
    # For each task, compute evidence and apply methods
    print("\n[3-5] Evidence-based methods (per-task):")
    print("  (This requires per-task support evidence computation)")
    print("  Use scripts/compute_evidence.py for detailed analysis")

    # Quick demo: equal-weight merge (alpha = 1/3 each)
    print("\n[Demo] Equal-weight linear merge (alpha=1/3 each):")
    equal_weights = [1.0 / 3] * 3
    merged_state = linear_merge(base_state, specialist_states, equal_weights)
    merged_model, _ = load_model_from_checkpoint(args.base_checkpoint, device)
    merged_model.load_state_dict(merged_state)
    merged_model.eval()
    evaluate_model(merged_model, device, tasks, "EqualMerge")


if __name__ == "__main__":
    main()
