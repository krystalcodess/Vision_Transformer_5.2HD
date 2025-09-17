import evaluate
import os
import random
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# ===== CONFIG =====
SUBSET_CLASSES = ["airplane", "ship",
                  "cat", "automobile", "dog"]
TRAIN_FRACTION = 0.75
MODEL_NAME = "google/vit-base-patch16-224"
OUTDIR = "outputs"
SEED = 42
IMG_SIZE = 224
BATCH = 64
EPOCHS = 15
LR = 5e-5
WD = 0.05
WARMUP_RATIO = 0.1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CIFAR-10 mapping
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
subset_to_idx = {c: CIFAR10_CLASSES.index(c) for c in SUBSET_CLASSES}
id2label = {i: c for i, c in enumerate(SUBSET_CLASSES)}
label2id = {c: i for i, c in id2label.items()}

# ===== DATASETS =====


class CIFARSubset(Dataset):
    def __init__(self, split="train", processor=None, train_fraction=1.0):
        self.processor = processor
        is_train = split == "train"

        tfm = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])

        base = datasets.CIFAR10(
            root="data", train=is_train, download=True, transform=tfm)

        # filter to subset classes
        idxs = [i for i, (img, label) in enumerate(
            base) if label in subset_to_idx.values()]

        # remap labels to 0..K-1 in subset order
        images, labels = [], []
        for i in idxs:
            img, y = base[i]
            cname = CIFAR10_CLASSES[y]
            images.append(img)
            labels.append(label2id[cname])

        if is_train and 0 < train_fraction < 1.0:
            _, keep = train_test_split(
                np.arange(len(images)),
                train_size=train_fraction,
                random_state=SEED,
                stratify=labels
            )
            images = [images[i] for i in keep]
            labels = [labels[i] for i in keep]

        # make a small val split from train if needed
        if split == "val":
            train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
                images, labels, test_size=0.2, random_state=SEED, stratify=labels
            )
            images, labels = val_imgs, val_lbls

        self.images = images
        self.labels = labels

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # convert tensor->PIL for processor
        img = transforms.ToPILImage()(img)
        enc = processor(images=img, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ===== PROCESSOR & MODEL =====
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(SUBSET_CLASSES),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# datasets
train_ds = CIFARSubset(split="train", processor=processor,
                       train_fraction=TRAIN_FRACTION)
val_ds = CIFARSubset(split="val",   processor=processor,
                     train_fraction=TRAIN_FRACTION)

# ===== METRICS =====
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": acc.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
    }


# ===== TRAINER =====
os.makedirs(os.path.join(OUTDIR, "ckpts"), exist_ok=True)
args = TrainingArguments(
    output_dir=os.path.join(OUTDIR, "ckpts"),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=WD,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=50,
    report_to="none",
    save_total_limit=2,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(os.path.join(OUTDIR, "ckpts", "best"))
processor.save_pretrained(os.path.join(OUTDIR, "ckpts", "best"))
print("✅ Training complete. Best checkpoint saved.")
