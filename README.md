# ViT on CIFAR-10 (5-class subset)

Fine-tuning `google/vit-base-patch16-224` on a 5-class subset of CIFAR-10 (airplane, ship, cat, automobile, dog) with training/evaluation scripts, attention visualizations, and ready-to-use figures and report.

## Highlights
- Simple end-to-end pipeline with Hugging Face Trainer (fp16 enabled)
- Evaluation artifacts: confusion matrix, classification report, predictions CSV
- Qualitative attention overlays (correct/wrong) and grid
- Learning curves and metrics table figures
- Report draft included

## Quickstart

### 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train
Runs for 15 epochs and saves checkpoints under `outputs/ckpts/` (best model in `outputs/ckpts/best`). CIFAR-10 is auto-downloaded to `data/`.
```bash
python src/train.py
```

Notes:
- Effective train size is ~6,250 images (see Splits below). Batch size 64 → ~98 steps/epoch, ~1,470 steps total.
- Seeds are fixed to 42 for numpy/torch/python.

### 3) Evaluate (metrics, confusion matrix, predictions)
Generates `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, and `predictions.csv`.
```bash
python src/eval.py \
  --ckpt outputs/ckpts/best \
  --subset "airplane,ship,cat,automobile,dog" \
  --img_size 224 \
  --batch 128 \
  --outdir outputs/figures \
  --preds_csv outputs/predictions.csv
```

### 4) Attention visualizations
Saves side-by-side original+overlay images under `outputs/figures/attention/` and a grid.
```bash
python src/viz_attention.py \
  --ckpt outputs/ckpts/best \
  --subset "airplane,ship,cat,automobile,dog" \
  --img_size 224 \
  --num 8 \
  --mode mixed \
  --outdir outputs/figures/attention
```
`--mode` can be `mixed`, `correct`, or `wrong`.

### 5) Learning curves + metrics table figures
Creates `outputs/figures/learning_curves.png` and `outputs/figures/metrics_table.png`.
```bash
python src/plot_and_tables.py
```

## Results (snapshot)
- Accuracy (5-class test): 0.983
- Macro-F1 (5-class test): 0.983
- See `outputs/figures/confusion_matrix.png`, `outputs/figures/metrics_table.png`, and attention images under `outputs/figures/attention/`.

## Data & splits
- Base dataset: CIFAR-10 (auto-downloaded via `torchvision.datasets.CIFAR10`).
- Classes used: `airplane, ship, cat, automobile, dog`.
- Splits (as implemented):
  - Train: ~6,250 images total (≈1,250/class)
  - Validation: ~1,000 images total (≈200/class) from 20% of the filtered test set
  - Test (final eval): 5,000 images total (1,000/class)

## Compute environment
- SLURM cluster; 1× NVIDIA A100 GPU; 4 vCPU; 32 GB RAM; Linux; fp16 enabled.
- Training schedule: 15 epochs, batch 64, LR 5e-5, WD 0.05, warmup ratio 0.1.

## SLURM usage (optional)
Edit `my_job.sh` as needed, then submit:
```bash
sbatch my_job.sh
```
The script requests `--gres=gpu:a100:1`, 32 GB RAM, 4 CPUs, 4 hours.

## Project structure
```
src/
  train.py            # fine-tuning pipeline
  eval.py             # metrics, confusion matrix, predictions
  viz_attention.py    # attention overlays + grid
  plot_and_tables.py  # learning curves + metrics table figures
outputs/
  ckpts/              # checkpoints (best/ and checkpoint-*/)
  figures/            # confusion_matrix.png, metrics.json, report artifacts
    attention/        # attention overlay images
  predictions.csv     # per-sample predictions (id,true_label,pred_label)
outputs/repor
