"""
Embedding Extractor — bridges the PyTorch CNN models (AlexNet / ResNet18 / VGG16)
with the R XGBoost pipeline.
---------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza (pipeline extension)

Description:
    Loads a backbone already trained by one of:
        multimodal_alexnet_patch_level.py
        multimodal_resnet18_patch_level.py
        multimodal_vgg16_patch_level.py
    (i.e. a `best_model.pt` + `class_map.csv` + `num_classes.txt` saved in
    --results_dir), re-creates the EXACT same train/val/test split used during
    training (same --data, --split, --split_seed), runs every image through the
    backbone only (no classifier head), and writes one CSV per split:

        embeddings_train.csv
        embeddings_val.csv
        embeddings_test.csv

    Each CSV has columns: class_name, label, feat_0000, feat_0001, ...
    These CSVs are plain tabular data — read them directly from R with
    read.csv() / readxl, exactly like the "Nucleus: ..." feature columns in
    the original XGBoost script, and feed them into xgboost as a multiclass
    problem (num_class = 21 for UCMerced).

Usage:
    # Defaults match the UCMerced setup used to train the CNN models
    python extract_embeddings.py --backbone alexnet  --results_dir /kaggle/working/results_alexnet
    python extract_embeddings.py --backbone resnet18 --results_dir /kaggle/working/results_resnet18
    python extract_embeddings.py --backbone vgg16    --results_dir /kaggle/working/results_vgg16

    # Custom dataset / split / output location
    python extract_embeddings.py --backbone resnet18 \
        --results_dir /kaggle/working/results_resnet18 \
        --data /path/to/data --split 0.5 0.15 0.35 --split_seed 42 \
        --output_dir /kaggle/working/embeddings_resnet18

Dependencies:
    torch>=2.1.0
    torchvision>=0.16.0
    pandas>=2.0.0
    pillow>=10.0.0
    tqdm>=4.66.0
"""

import argparse
import os
import random
import sys

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    AlexNet_Weights, ResNet18_Weights, VGG16_Weights,
    alexnet, resnet18, vgg16,
)
from tqdm import tqdm

DEFAULT_DATA_PATH = "/kaggle/input/datasets/eman12345nasser/ucmerced-landuse/Images"
DEFAULT_SPLIT     = [0.5, 0.15, 0.35]


# ===============================================================
# Dataset (same folder-scanning / pre-split logic as the training scripts)
# ===============================================================
class ImageOnlyDataset(Dataset):
    """Minimal dataset that just loads an image + its integer label."""

    def __init__(self, items: list[dict], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]
        image = Image.open(sample["patch_path"]).convert("RGB")
        image = self.transform(image)
        return image, sample["label"], sample["class_name"]


def split_dataset(data_path: str, fractions: list[float], seed: int = 42):
    """Reproduces the same non-overlapping train/val/test split used during
    training, provided the same --data / --split / --split_seed are passed."""
    if not (1 <= len(fractions) <= 3):
        sys.exit("[ERROR] --split accepts 1, 2, or 3 values (e.g. 0.5 0.15 0.35)")
    total = sum(fractions)
    if total > 1.0 + 1e-6:
        sys.exit(f"[ERROR] --split values sum to {total:.4f} — must be ≤ 1.0")

    n_fracs = len(fractions)
    f_train = fractions[0]
    f_val   = fractions[1] if n_fracs >= 2 else 0.0
    f_test  = fractions[2] if n_fracs == 3 else 0.0
    want_val  = n_fracs >= 2
    want_test = n_fracs == 3

    class_names = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    )
    if not class_names:
        sys.exit(f"[ERROR] No class sub-folders found in '{data_path}'")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    _IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    train_items, val_items, test_items = [], ([] if want_val else None), ([] if want_test else None)
    rng = random.Random(seed)

    for class_name in class_names:
        label = class_to_idx[class_name]
        class_dir = os.path.join(data_path, class_name)
        files = [
            {"patch_path": os.path.join(class_dir, f), "label": label, "class_name": class_name}
            for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXT
        ]
        rng.shuffle(files)

        n = len(files)
        n_train = round(n * f_train)
        n_val   = min(round(n * f_val), n - n_train) if want_val else 0
        n_test  = (n - n_train - n_val) if want_test else 0

        train_items.extend(files[:n_train])
        if want_val:
            val_items.extend(files[n_train:n_train + n_val])
        if want_test:
            test_items.extend(files[n_train + n_val:n_train + n_val + n_test])

    return train_items, val_items, test_items, class_to_idx


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ===============================================================
# Backbone builders — mirror the Multimodal* model classes, embedding-only
# (clinical_input_dim=0, classifier head dropped) so state_dict keys line up
# with what best_model.pt was saved from.
# ===============================================================
def build_backbone_shell(name: str, num_classes: int) -> nn.Module:
    """
    Rebuilds the exact same nn.Module tree used at training time
    (backbone + classifier, clinical branch skipped since clinical_input_dim=0)
    so `load_state_dict` matches perfectly. Returns the FULL model; caller
    should use `.backbone(image)` to get the embedding, ignoring `.classifier`.
    """
    if name == "alexnet":
        backbone = alexnet(weights=AlexNet_Weights.DEFAULT)
        backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])
        feature_dim = 4096
    elif name == "vgg16":
        backbone = vgg16(weights=VGG16_Weights.DEFAULT)
        backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])
        feature_dim = 4096
    elif name == "resnet18":
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        sys.exit(f"[ERROR] Unknown --backbone '{name}'")

    class _Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.image_dropout = nn.Dropout(0.5)
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.7),
                nn.Linear(512, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.image_dropout(self.backbone(x)))

    return _Shell()


# ===============================================================
# Extraction
# ===============================================================
def extract_split(model, items, transform, device, batch_size, split_name) -> pd.DataFrame:
    if not items:
        print(f"[INFO] No items for split '{split_name}' — skipping.")
        return pd.DataFrame()

    loader = DataLoader(
        ImageOnlyDataset(items, transform),
        batch_size=batch_size, shuffle=False, num_workers=4,
    )

    rows = []
    model.eval()
    with torch.no_grad():
        for imgs, labels, class_names in tqdm(loader, desc=f"Extracting [{split_name}]"):
            imgs = imgs.to(device)
            feats = model.backbone(imgs)          # embedding only, classifier head unused
            feats = feats.cpu().numpy()
            for feat_vec, label, cname in zip(feats, labels.numpy(), class_names):
                row = {"class_name": cname, "label": int(label)}
                row.update({f"feat_{i:04d}": v for i, v in enumerate(feat_vec)})
                rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Extract CNN embeddings (AlexNet/ResNet18/VGG16) as tabular CSVs for XGBoost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone", type=str, required=True,
                         choices=["alexnet", "resnet18", "vgg16"])
    parser.add_argument("--results_dir", type=str, required=True,
                         help="Directory containing best_model.pt, class_map.csv, num_classes.txt "
                              "(produced by the matching training script).")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=float, nargs="+", default=DEFAULT_SPLIT)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Where to save embeddings_{train,val,test}.csv. "
                              "Defaults to <results_dir>/embeddings")
    parser.add_argument("--batch_size", type=int, default=64)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring unrecognized args: {unknown}")

    output_dir = args.output_dir or os.path.join(args.results_dir, "embeddings")
    os.makedirs(output_dir, exist_ok=True)

    num_classes_path = os.path.join(args.results_dir, "num_classes.txt")
    if not os.path.isfile(num_classes_path):
        sys.exit(f"[ERROR] '{num_classes_path}' not found. Train the {args.backbone} "
                  f"model first (it writes this file automatically).")
    num_classes = int(open(num_classes_path).read().strip())

    model_path = os.path.join(args.results_dir, "best_model.pt")
    if not os.path.isfile(model_path):
        sys.exit(f"[ERROR] '{model_path}' not found. Train the {args.backbone} model first.")

    print(f"[INFO] Rebuilding '{args.backbone}' shell with num_classes={num_classes} "
          f"and loading weights from '{model_path}'")
    model = build_backbone_shell(args.backbone, num_classes)
    state_dict = torch.load(model_path, map_location="cpu")
    # Strip potential "module." prefix left by nn.DataParallel checkpoints
    state_dict = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"[INFO] Re-creating the train/val/test split from '{args.data}' "
          f"(split={args.split}, seed={args.split_seed}) to match training exactly.")
    train_items, val_items, test_items, class_to_idx = split_dataset(
        args.data, args.split, seed=args.split_seed
    )

    transform = build_eval_transform()

    for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        df = extract_split(model, items, transform, device, args.batch_size, split_name)
        if df.empty:
            continue
        out_path = os.path.join(output_dir, f"embeddings_{split_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"[INFO] Saved {len(df)} rows -> '{out_path}'")

    print(f"\nDone. Feature dimensionality: "
          f"{4096 if args.backbone in ('alexnet', 'vgg16') else 512}")
    print(f"Class map (label -> class_name): "
          f"{ {v: k for k, v in class_to_idx.items()} }")


if __name__ == "__main__":
    main()
