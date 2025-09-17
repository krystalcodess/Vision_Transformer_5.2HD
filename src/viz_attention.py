# src/viz_attention.py
import os, argparse, math, numpy as np, torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, ViTForImageClassification

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/ckpts/best",
                   help="Checkpoint dir (contains config.json, pytorch_model.bin, etc.)")
    p.add_argument("--subset", type=str, default="airplane,ship,cat,automobile,dog",
                   help="Comma-separated CIFAR-10 classes to visualize")
    p.add_argument("--img_size", type=int, default=224, help="Model input size")
    p.add_argument("--num", type=int, default=8, help="Number of samples to visualize")
    p.add_argument("--mode", type=str, default="mixed", choices=["mixed","correct","wrong"],
                   help="Choose correct-only, wrong-only, or mixed examples")
    p.add_argument("--outdir", type=str, default="outputs/figures/attention",
                   help="Where to save overlays")
    p.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def attention_rollout(attentions: list):
    """
    attentions: list of tensors [L x (B, H, T, T)] from ViT forward with output_attentions=True
    Returns: rollout attention from CLS to patch tokens, shape [B, T-1]
    """
    # Stack layers, average heads
    # out: [L, B, T, T]
    layers = torch.stack([a.mean(dim=1) for a in attentions], dim=0).squeeze(2)
    # Add identity (residual) and normalize rows
    eye = torch.eye(layers.size(-1), device=layers.device).unsqueeze(0).unsqueeze(1)  # [1,1,T,T]
    layers = layers + eye  # broadcast over L,B
    layers = layers / layers.sum(dim=-1, keepdim=True)
    # Cumulative product across layers
    rollout = layers[0]
    for i in range(1, layers.size(0)):
        rollout = rollout @ layers[i]
    # Take CLS row (index 0) attention to all tokens
    cls_to_tokens = rollout[:, 0, 1:]  # drop CLS->CLS, return [B, T-1]
    return cls_to_tokens

def make_overlay(img_np, att_map, alpha=0.45):
    """Blend heatmap (0..1) over RGB image (H,W,3)"""
    h, w = img_np.shape[:2]
    # Normalize heatmap
    hm = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-6)
    hm = np.clip(hm, 0, 1)
    # Resize heatmap to image size (nearest-neighbor keeps blocky patch look)
    hm = np.array(Image.fromarray((hm*255).astype(np.uint8)).resize((w, h), Image.NEAREST)) / 255.0
    # Simple overlay using a colormap-like grayscale blend
    overlay = img_np.astype(np.float32) / 255.0
    overlay = (1 - alpha) * overlay + alpha * hm[..., None]
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    subset_classes = [c.strip() for c in args.subset.split(",")]
    cifar10_classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    keep_idx = [cifar10_classes.index(c) for c in subset_classes]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model & processor with attentions
    processor = AutoImageProcessor.from_pretrained(args.ckpt)
    model = ViTForImageClassification.from_pretrained(args.ckpt, output_attentions=True).to(device).eval()

    # Build filtered CIFAR-10 test set as PIL images
    test_raw = datasets.CIFAR10(root="data", train=False, download=True)

    resize = transforms.Resize((args.img_size, args.img_size))

    pil_images, y_true, orig_ids = [], [], []
    for i in range(len(test_raw)):
        img, y = test_raw[i]
        if y in keep_idx:
            pil_images.append(resize(img))  # resized PIL
            y_true.append(keep_idx.index(y))  # remapped label 0..K-1
            orig_ids.append(i)

    # Run a quick pass to get predictions for selection (correct/wrong/mixed)
    preds = []
    with torch.no_grad():
        for i in range(0, len(pil_images), 128):
            batch = pil_images[i:i+128]
            enc = processor(images=batch, return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=1).cpu().numpy())
    preds = np.array(preds)
    y_true = np.array(y_true)

    # Select indices based on mode
    correct_idx = np.where(preds == y_true)[0]
    wrong_idx   = np.where(preds != y_true)[0]
    pool = None
    if args.mode == "correct":
        pool = correct_idx
    elif args.mode == "wrong":
        pool = wrong_idx
    else:
        # mixed: half correct, half wrong if possible
        n_half = args.num // 2
        take_c = rng.choice(correct_idx, size=min(n_half, len(correct_idx)), replace=False) if len(correct_idx) else np.array([],dtype=int)
        take_w = rng.choice(wrong_idx,   size=args.num - len(take_c), replace=False) if len(wrong_idx) else np.array([],dtype=int)
        sel = np.concatenate([take_c, take_w])
        rng.shuffle(sel)
        pool = sel

    if len(pool) == 0:
        print(f"No samples available for mode='{args.mode}'. Falling back to random selection.")
        pool = rng.choice(np.arange(len(pil_images)), size=min(args.num, len(pil_images)), replace=False)

    if len(pool) > args.num:
        pool = rng.choice(pool, size=args.num, replace=False)

    # Generate overlays
    saved_paths = []
    for rank, idx in enumerate(pool):
        img_pil = pil_images[idx]
        enc = processor(images=img_pil, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model(**enc)
        att_tokens = attention_rollout(out.attentions)  # [B, T-1]
        att = att_tokens[0].detach().cpu().numpy()

        # Infer patch grid from token count
        num_patches = att.shape[0]
        grid = int(math.sqrt(num_patches))
        att_grid = att.reshape(grid, grid)

        # Prepare overlay
        img_np = np.array(img_pil.convert("RGB"))
        overlay = make_overlay(img_np, att_grid, alpha=args.alpha)

        true_lbl = subset_classes[y_true[idx]]
        pred_lbl = subset_classes[preds[idx]]
        status = "correct" if y_true[idx] == preds[idx] else "wrong"
        out_path = os.path.join(args.outdir, f"att_{rank:02d}_{status}_true-{true_lbl}_pred-{pred_lbl}.png")

        # Save side-by-side original and overlay
        canvas = Image.new("RGB", (img_np.shape[1]*2, img_np.shape[0]))
        canvas.paste(Image.fromarray(img_np), (0,0))
        canvas.paste(Image.fromarray(overlay), (img_np.shape[1],0))
        canvas.save(out_path)
        saved_paths.append(out_path)
        print(f"Saved → {out_path}")

    # Also save a quick contact sheet grid for the overlays only
    if saved_paths:
        imgs = [Image.open(p).crop((img_np.shape[1],0,img_np.shape[1]*2,img_np.shape[0])) for p in saved_paths]  # right half (overlay)
        cols = min(4, len(imgs))
        rows = int(math.ceil(len(imgs)/cols))
        w, h = imgs[0].size
        sheet = Image.new("RGB", (cols*w, rows*h), (255,255,255))
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            sheet.paste(im, (c*w, r*h))
        sheet_path = os.path.join(args.outdir, f"attention_grid_{args.mode}.png")
        sheet.save(sheet_path)
        print(f"Saved grid → {sheet_path}")

if __name__ == "__main__":
    main()

