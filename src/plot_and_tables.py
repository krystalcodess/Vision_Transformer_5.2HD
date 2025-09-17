import os
import json
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


REPO_ROOT = "/home/s223212228/vit-cifar10"
TRAINER_STATE = os.path.join(REPO_ROOT, "outputs/ckpts/checkpoint-1470/trainer_state.json")
REPORT_DIR = os.path.join(REPO_ROOT, "outputs/figures")
LEARNING_CURVES_PNG = os.path.join(REPORT_DIR, "learning_curves.png")
CLASSIF_REPORT_TXT = os.path.join(REPO_ROOT, "outputs/figures/classification_report.txt")
METRICS_TABLE_PNG = os.path.join(REPORT_DIR, "metrics_table.png")


def load_trainer_history(path: str):
    with open(path, "r") as f:
        state = json.load(f)
    history: List[dict] = state.get("log_history", [])
    # Training loss entries have keys {'loss','step',...}
    train_losses: List[Tuple[int, float]] = []
    eval_points: List[Tuple[float, float, float]] = []  # (epoch, eval_accuracy, eval_loss)

    for item in history:
        if "loss" in item and "step" in item and "eval_loss" not in item:
            step = int(item.get("step"))
            loss = float(item.get("loss"))
            train_losses.append((step, loss))
        if "eval_accuracy" in item:
            epoch = float(item.get("epoch", 0.0))
            acc = float(item.get("eval_accuracy"))
            evl = float(item.get("eval_loss", 0.0))
            eval_points.append((epoch, acc, evl))

    train_losses.sort(key=lambda x: x[0])
    eval_points.sort(key=lambda x: x[0])
    return train_losses, eval_points


def plot_learning_curves():
    os.makedirs(REPORT_DIR, exist_ok=True)
    train_losses, eval_points = load_trainer_history(TRAINER_STATE)

    steps = [s for s, _ in train_losses]
    losses = [l for _, l in train_losses]

    epochs = [e for e, _, _ in eval_points]
    eval_acc = [a for _, a, _ in eval_points]
    eval_loss = [l for _, _, l in eval_points]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Training loss vs steps
    axes[0].plot(steps, losses, color="#1f77b4")
    axes[0].set_title("Training loss vs steps")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Eval accuracy vs epochs
    axes[1].plot(epochs, eval_acc, marker="o", color="#2ca02c")
    axes[1].set_title("Eval accuracy vs epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].grid(True, alpha=0.3)

    # Eval loss vs epochs
    axes[2].plot(epochs, eval_loss, marker="o", color="#d62728")
    axes[2].set_title("Eval loss vs epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(LEARNING_CURVES_PNG, dpi=200)
    plt.close(fig)
    print(f"Saved learning curves → {LEARNING_CURVES_PNG}")


def parse_classification_report(text: str):
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Pattern: class precision recall f1 support
    row_re = re.compile(r"^\s*(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)\s*$")
    rows = []
    for ln in lines:
        if not ln.strip():
            continue
        if ln.strip().startswith("accuracy"):
            continue
        m = row_re.match(ln)
        if m:
            cls, p, r, f1, sup = m.groups()
            rows.append([cls, float(p), float(r), float(f1), int(sup)])
    return rows


def render_metrics_table():
    with open(CLASSIF_REPORT_TXT, "r") as f:
        txt = f.read()
    rows = parse_classification_report(txt)

    # Separate class rows and summary rows (macro avg, weighted avg)
    class_rows = [r for r in rows if r[0] not in {"macro", "macro avg", "weighted", "weighted avg"}]
    summary_rows = [r for r in rows if r[0] in {"macro", "macro avg", "weighted", "weighted avg"}]

    headers = ["Class", "Precision", "Recall", "F1-score", "Support"]
    cell_text = []
    for r in class_rows + summary_rows:
        cls, p, rcl, f1, sup = r
        name = str(cls)
        cell_text.append([
            name,
            f"{p:.4f}",
            f"{rcl:.4f}",
            f"{f1:.4f}",
            str(sup),
        ])

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(cell_text) + 1)))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title("Classification metrics", pad=10)
    plt.tight_layout()
    plt.savefig(METRICS_TABLE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics table → {METRICS_TABLE_PNG}")


if __name__ == "__main__":
    plot_learning_curves()
    render_metrics_table()


