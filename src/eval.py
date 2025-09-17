# src/eval.py
import argparse, os, json, numpy as np, torch
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import csv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/ckpts/best",
                   help="Path to checkpoint dir (contains config.json, pytorch_model.bin, etc.)")
    p.add_argument("--subset", type=str, default="airplane,ship,cat,automobile,dog",
                   help="Comma-separated CIFAR-10 class names to evaluate")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--outdir", type=str, default="outputs/figures")
    p.add_argument("--preds_csv", type=str, default="outputs/predictions.csv")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.preds_csv), exist_ok=True)

    subset_classes = [c.strip() for c in args.subset.split(",")]
    cifar10_classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    keep_idxs = [cifar10_classes.index(c) for c in subset_classes]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor & model from checkpoint
    processor = AutoImageProcessor.from_pretrained(args.ckpt)
    model = ViTForImageClassification.from_pretrained(args.ckpt).to(device).eval()

    # Build test set filtered to subset
    tfm = transforms.Compose([transforms.Resize((args.img_size, args.img_size))])
    testset = datasets.CIFAR10(root="data", train=False, download=True)
    # Collect images/labels only for chosen classes
    pil_images, labels, filenames = [], [], []
    for i in range(len(testset)):
        img, y = testset[i]
        if y in keep_idxs:
            pil_images.append(img)  # PIL already
            labels.append(keep_idxs.index(y))  # remap to 0..K-1
            # CIFAR dataset has no filenames; synthesize stable IDs
            filenames.append(f"test_{i:05d}.png")

    # Batched inference
    preds = []
    with torch.no_grad():
        for i in range(0, len(pil_images), args.batch):
            batch_pil = pil_images[i:i+args.batch]
            # Ensure resized before processor (processor can resize too; we keep tfm for determinism)
            batch_pil = [tfm(im) if isinstance(im, Image.Image) else im for im in batch_pil]
            enc = processor(images=batch_pil, return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=1).cpu().numpy())

    y_true = np.array(labels)
    y_pred = np.array(preds)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    report_str = classification_report(y_true, y_pred, target_names=subset_classes, digits=4)
    print("\n=== Classification Report ===\n")
    print(report_str)
    print(f"\nAccuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")

    # Save metrics
    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "macro_f1": f1m,
                   "classes": subset_classes}, f, indent=2)
    # Save classification report text
    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(report_str)

    # Confusion matrix figure
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(subset_classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=subset_classes)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, values_format="d", xticks_rotation=45, cmap=None)  # default cmap
    plt.tight_layout()
    cm_path = os.path.join(args.outdir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix → {cm_path}")

    # Save per-sample predictions CSV
    with open(args.preds_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id","true_label","pred_label"])
        for fn, t, p in zip(filenames, y_true, y_pred):
            writer.writerow([fn, subset_classes[t], subset_classes[p]])
    print(f"Saved predictions CSV → {args.preds_csv}")

if __name__ == "__main__":
    main()
