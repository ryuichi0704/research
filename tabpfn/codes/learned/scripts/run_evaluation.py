"""Full evaluation: base vs specialists vs evidence-weighted portfolio vs merge.

Compares on OpenML classification tasks:
1. Broad base PFN (pretrained)
2. SCM specialist
3. Tree specialist
4. Soft evidence-weighted portfolio (weighted avg of specialist predictions)
5. Hard selection (best specialist per task)
6. Linear support-adaptive merge (weight-merged checkpoint)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

import openml
from openml.tasks import TaskType

from tfmplayground.interface import NanoTabPFNClassifier, get_feature_preprocessor
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.evaluation import TOY_TASKS_CLASSIFICATION, TABARENA_TASKS

from src.utils import load_model_from_checkpoint
from src.merging import linear_merge, checkpoint_distance


FAMILY_NAMES = ["SCM", "Tree"]


def load_all_models(base_path, specialist_paths, device):
    """Load base and specialist models."""
    base_model, base_ckpt = load_model_from_checkpoint(base_path, device)
    base_state = base_ckpt["model"]

    specialists = []
    specialist_states = []
    for path in specialist_paths:
        model, ckpt = load_model_from_checkpoint(path, device)
        specialists.append(model)
        specialist_states.append(ckpt["model"])

    return base_model, specialists, base_state, specialist_states


def compute_prequential_log_loss_fast(model, X_train, y_train, device, max_prefix=50, min_ctx=2):
    """Compute prequential log loss using incremental prefixes.

    For efficiency, evaluate at most max_prefix points.
    Starts from min_ctx context points to avoid NaN from std computation.
    """
    model.eval()
    n = min(len(X_train), max_prefix)
    if n <= min_ctx:
        return 0.0  # not enough data
    cum_loss = 0.0

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    with torch.no_grad():
        for i in range(min_ctx, n):
            x_all = X_t[: i + 1].unsqueeze(0).to(device)  # (1, i+1, d)
            y_ctx = y_t[:i].unsqueeze(0).to(device)        # (1, i)

            logits = model((x_all, y_ctx), single_eval_pos=i)  # (1, 1, C)
            probs = F.softmax(logits[0, 0], dim=-1)

            true_label = int(y_train[i])
            if true_label < probs.shape[0]:
                p = probs[true_label].item()
            else:
                p = 1e-10
            cum_loss += -np.log(max(p, 1e-10))

    return cum_loss


def compute_evidence_weights(specialists, X_train, y_train, device, prior_weights=None):
    """Compute posterior weights over specialists."""
    from scipy.special import logsumexp

    M = len(specialists)
    if prior_weights is None:
        prior_weights = [1.0 / M] * M

    scores = np.zeros(M)
    for j, (model, rho) in enumerate(zip(specialists, prior_weights)):
        pll = compute_prequential_log_loss_fast(model, X_train, y_train, device)
        scores[j] = np.log(max(rho, 1e-30)) - pll

    weights = np.exp(scores - logsumexp(scores))
    return weights


def evaluate_on_task(task_id, base_model, specialists, base_state, specialist_states, device):
    """Evaluate all methods on a single OpenML task. Returns dict of method->accuracy."""
    task = openml.tasks.get_task(task_id, download_splits=False)
    if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
        return None

    dataset = task.get_dataset(download_data=False)
    n_features = dataset.qualities["NumberOfFeatures"]
    n_samples = dataset.qualities["NumberOfInstances"]
    if n_features > 500 or n_samples > 10_000:
        return None

    X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    train_idx, test_idx = task.get_train_test_split_indices(fold=0, repeat=0)

    X_train = X.iloc[train_idx].to_numpy()
    y_train_raw = y.iloc[train_idx].to_numpy()
    X_test = X.iloc[test_idx].to_numpy()
    y_test_raw = y.iloc[test_idx].to_numpy()

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    # Preprocess features
    preprocessor = get_feature_preprocessor(X_train)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    results = {}

    # Helper: get predictions from a model
    def predict_with_model(model):
        clf = NanoTabPFNClassifier(model=model, device=device, num_mem_chunks=64)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        return y_pred, y_proba

    # 1. Base
    y_pred, _ = predict_with_model(base_model)
    results["Base"] = balanced_accuracy_score(y_test, y_pred)

    # 2. Individual specialists
    specialist_probas = []
    for name, model in zip(FAMILY_NAMES, specialists):
        y_pred, y_proba = predict_with_model(model)
        results[f"Specialist-{name}"] = balanced_accuracy_score(y_test, y_pred)
        specialist_probas.append(y_proba)

    # 3. Compute evidence weights
    weights = compute_evidence_weights(specialists, X_train_p, y_train, device)
    results["Evidence_weights"] = {name: f"{w:.3f}" for name, w in zip(FAMILY_NAMES, weights)}

    # 4. Soft portfolio (weighted average of predictions)
    n_classes = max(p.shape[1] for p in specialist_probas)
    # Pad probas to same number of classes
    padded_probas = []
    for p in specialist_probas:
        if p.shape[1] < n_classes:
            pad = np.zeros((p.shape[0], n_classes - p.shape[1]))
            p = np.concatenate([p, pad], axis=1)
        padded_probas.append(p)

    soft_proba = sum(w * p for w, p in zip(weights, padded_probas))
    soft_pred = soft_proba.argmax(axis=1)
    results["Soft-Portfolio"] = balanced_accuracy_score(y_test, soft_pred)

    # 5. Hard selection
    best_j = np.argmax(weights)
    results["Hard-Selection"] = results[f"Specialist-{FAMILY_NAMES[best_j]}"]

    # 6. Linear merge
    merged_state = linear_merge(base_state, specialist_states, weights.tolist())
    merged_model, _ = load_model_from_checkpoint(
        "checkpoints/nanotabpfn.pth", device
    )
    merged_model.load_state_dict(merged_state)
    merged_model.eval()
    y_pred, _ = predict_with_model(merged_model)
    results["Linear-Merge"] = balanced_accuracy_score(y_test, y_pred)

    return dataset.name, results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_path = "checkpoints/nanotabpfn.pth"
    specialist_paths = [
        "workdir/specialist_scm/latest_checkpoint.pth",
        "workdir/specialist_tree/latest_checkpoint.pth",
    ]

    base_model, specialists, base_state, specialist_states = load_all_models(
        base_path, specialist_paths, device
    )

    # Print checkpoint distances
    for name, ss in zip(FAMILY_NAMES, specialist_states):
        dist = checkpoint_distance(base_state, ss)
        print(f"  ||theta_{name} - theta_0|| = {dist:.4f}")

    tasks = TABARENA_TASKS
    print(f"\nEvaluating on {len(tasks)} TABARENA tasks...")
    print("=" * 80)

    all_results = {}
    for task_id in tasks:
        try:
            torch.cuda.empty_cache()
            result = evaluate_on_task(
                task_id, base_model, specialists, base_state, specialist_states, device
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"\n--- task {task_id}: SKIPPED ({e.__class__.__name__}) ---")
            torch.cuda.empty_cache()
            continue
        if result is None:
            continue
        ds_name, metrics = result
        all_results[ds_name] = metrics

        print(f"\n--- {ds_name} ---")
        for method, value in metrics.items():
            if method == "Evidence_weights":
                print(f"  Evidence weights: {value}")
            else:
                print(f"  {method:20s}: {value:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY (avg balanced accuracy)")
    print("=" * 80)
    methods = ["Base", "Specialist-SCM", "Specialist-Tree",
               "Soft-Portfolio", "Hard-Selection", "Linear-Merge"]
    for method in methods:
        scores = [r[method] for r in all_results.values() if method in r]
        if scores:
            print(f"  {method:20s}: {np.mean(scores):.4f} (n={len(scores)})")


if __name__ == "__main__":
    main()
