"""
Multimodal ResNet18 Classifier for Reactive Follicular Hyperplasia and Follicular Lymphoma
---------------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Description:
    This script implements a multimodal deep learning pipeline that integrates
    histopathological image patches, clinicopathologic and nuclear morphometric features to classify
    lymphoid lesions into Reactive Follicular Hyperplasia (RFH) and Follicular Lymphoma (FL).

    The model is based on a ResNet18 convolutional backbone for image embeddings,
    combined with a fully-connected network for clinical and nuclear morphometric data.
    The network is trained and validated on labeled patch-level data, and evaluated on
    an external test set using standard classification metrics and explainable outputs.

    NOTE: This version keeps the dataset defaults from the original UCMerced-based
    script (auto-split from a single root folder, no CLI args required to run it
    directly inside a Kaggle notebook cell), while adopting the more complete
    train/val/test pipeline, augmentation, TensorBoard logging, and class-map
    persistence used in the AlexNet version of this script.

Usage:
    # Just run it directly (Kaggle notebook / no CLI args) — uses the defaults below:
    #   --data   = /kaggle/input/datasets/eman12345nasser/ucmerced-landuse/Images
    #   --split  = 0.5 0.15 0.35   (train / val / test)
    #   --mode   = train
    python multimodal_resnet18_patch_level.py

    # ── Auto-split from a single root folder with augmentation ──
    python multimodal_resnet18_patch_level.py --data data/ --split 0.7 0.2 0.1 --mode train --augment

    # 80% train / 20% val, no test split
    python multimodal_resnet18_patch_level.py --data data/ --split 0.8 0.2 --mode train

    # 100% goes to the chosen mode — useful for standalone eval after training
    python multimodal_resnet18_patch_level.py --data data/ --split 1.0 --mode test

    # ── Pre-split folders ──
    python multimodal_resnet18_patch_level.py --mode train --train_dir data/train --augment
    python multimodal_resnet18_patch_level.py --mode train --train_dir data/train --val_dir data/val
    python multimodal_resnet18_patch_level.py --mode val   --val_dir  data/val
    python multimodal_resnet18_patch_level.py --mode test  --test_dir data/test

Dependencies:
    torch>=2.1.0
    torchvision>=0.16.0
    pandas>=2.0.0
    numpy>=1.24.0
    matplotlib>=3.8.0
    seaborn>=0.13.0
    scikit-learn>=1.3.0
    pillow>=10.0.0
    tqdm>=4.66.0
    openpyxl>=3.1.0
    tensorboard>=2.14.0
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm
from torchvision.transforms import v2

# ===============================================================
# Default paths (kept from the original UCMerced-based script so this
# file can be run with zero CLI arguments, e.g. straight from a Kaggle cell)
# ===============================================================
DEFAULT_DATA_PATH    = "/kaggle/input/datasets/eman12345nasser/ucmerced-landuse/Images"
DEFAULT_RESULTS_DIR  = "/kaggle/working/results"
DEFAULT_SPLIT        = [0.5, 0.15, 0.35]   # train / val / test, matches the original script


# ===============================================================
# Dataset Definition
# ===============================================================
class MultimodalDataset(Dataset):
    """
    Folder-based dataset that accepts any class folder name (string or integer).

    Expected structure:
        image_dir/
            class_a/            <- class name (any string)
                img1.jpg
                img2.jpg
            class_b/            <- class name (any string)
                img3.jpg

    Class names are sorted alphabetically and assigned integer labels
    starting from 0. The mapping is stored in ``self.class_to_idx`` and
    printed on load so you always know which folder maps to which label.
    """

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        image_dir: str | None,
        transform=None,
        class_to_idx: dict | None = None,
        items: list[dict] | None = None,
    ):
        """
        Args:
            image_dir:     Root directory whose sub-folders are class names.
                           Pass ``None`` when supplying ``items`` directly
                           (auto-split mode).
            transform:     Torchvision transform applied to each image.
            class_to_idx:  Optional fixed mapping {class_name: int_label}.
                           Pass the training set's mapping to val/test so
                           labels are always consistent.
            items:         Pre-built list of dicts with keys ``patch_path``,
                           ``label``, ``class_name``.  When provided,
                           ``image_dir`` scanning is skipped entirely.
        """
        self.image_dir     = image_dir
        self.transform     = transform
        self.clinical_cols: list[str] = []

        # ── Mode A: pre-split item list supplied directly ──────────
        if items is not None:
            self.items        = items
            self.class_to_idx = class_to_idx or {}
            counts = {}
            for item in self.items:
                counts[item["class_name"]] = counts.get(item["class_name"], 0) + 1
            print(f"\nDataset from pre-split list — {len(self.items)} images")
            print("  Class map:")
            for name, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
                n = counts.get(name, 0)
                print(f"    [{idx}] {name:>20s}  —  {n} images")
            return

        # ── Mode B: scan image_dir (original folder-based behaviour) ─
        if image_dir is None:
            raise ValueError("Either image_dir or items must be provided.")

        self.items: list[dict] = []

        # Discover class folders (any name, sorted for reproducibility)
        class_names = sorted(
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        )

        if not class_names:
            raise ValueError(
                f"No sub-folders found in '{image_dir}'. "
                "Each class must be its own folder."
            )

        # Build or reuse the class -> label mapping
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Scan images
        for class_name in class_names:
            if class_name not in self.class_to_idx:
                print(f"[WARNING] Folder '{class_name}' not in class_to_idx — skipped.")
                continue
            label     = self.class_to_idx[class_name]
            class_dir = os.path.join(image_dir, class_name)
            if os.path.isdir(class_dir) and len(os.listdir(class_dir)) > 0:
                for f in os.listdir(class_dir):
                    if os.path.splitext(f)[1].lower() in self._IMAGE_EXTENSIONS:
                        self.items.append(
                            {
                                "patch_path": os.path.join(class_dir, f),
                                "label":      label,
                                "class_name": class_name,
                            }
                        )

        # Summary
        counts = {}
        for item in self.items:
            counts[item["class_name"]] = counts.get(item["class_name"], 0) + 1
        print(f"\nLoaded {len(self.items)} images from '{image_dir}'")
        print("  Class map:")
        for name, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
            n = counts.get(name, 0)
            print(f"    [{idx}] {name:>20s}  —  {n} images")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample = self.items[idx]
        image = Image.open(sample["patch_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Empty clinical tensor — model handles clinical_input_dim=0
        clinical = torch.zeros(0, dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return image, clinical, label


# ===============================================================
# Model Definition (ResNet18 + Clinical Branch)
# ===============================================================
class MultimodalResNet18(nn.Module):
    """
    ResNet18 image backbone optionally fused with a clinical/morphometric branch.

    When ``clinical_input_dim == 0`` the clinical branch is skipped entirely
    and the model behaves as a pure image classifier. ``num_classes`` is
    auto-detected from the dataset's class folders (e.g. 21 for the UCMerced
    Land Use dataset), so it never needs to be hard-coded.
    """

    def __init__(self, clinical_input_dim: int, num_classes: int = 21):
        super().__init__()

        # Load ResNet18 with up-to-date weights API (pretrained=True is deprecated)
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Drop the final fc layer so we get raw embeddings, keep its input size.
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = feature_dim

        self.image_dropout = nn.Dropout(0.5)

        # Clinical branch — optional
        self.use_clinical = clinical_input_dim > 0
        if self.use_clinical:
            self.clinical_net = nn.Sequential(
                nn.Linear(clinical_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            fusion_dim = self.feature_dim + 32
        else:
            self.clinical_net = None
            fusion_dim = self.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes),
        )

    def forward(self, image: torch.Tensor, clinical_data: torch.Tensor) -> torch.Tensor:
        x = self.backbone(image)          # (B, feature_dim)
        x = self.image_dropout(x)

        if self.use_clinical and clinical_data.shape[-1] > 0:
            clinical_features = self.clinical_net(clinical_data)
            x = torch.cat((x, clinical_features), dim=1)

        return self.classifier(x)


# ===============================================================
# Auto-split helper
# ===============================================================
def split_dataset(
    data_path: str,
    fractions: list[float],
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """
    Collect all images under *data_path* (class-folder structure) and
    randomly split them into up to three **non-overlapping** partitions:
    train / val / test.

    Guarantees
    ----------
    * Every image appears in exactly one partition — no leakage between splits.
    * ``round()`` on individual fractions can make the per-class counts sum to
      more than ``n``.  We clamp ``n_val`` so that ``n_train + n_val <= n``
      before computing the test remainder, so no images are silently dropped
      and ``n_test`` is never negative.
    * When only 1 or 2 fractions are supplied the unused partition lists are
      set to ``None`` (not ``[]``) so callers can distinguish "intentionally
      no test split" from "split produced zero test images by accident".

    Args:
        data_path:  Root directory with one sub-folder per class.
        fractions:  1, 2, or 3 floats that must sum to ≤ 1.0.
                    • [0.7]          → 70 % train, val=None,  test=None
                    • [0.7, 0.2]     → 70 % train, 20 % val,  test=None
                    • [0.7, 0.2, 0.1]→ 70 % train, 20 % val,  10 % test
        seed:       Random seed for reproducibility.

    Returns:
        (train_items, val_items, test_items, class_to_idx)
        val_items / test_items are ``None`` when that split was not requested.
        Each item list is a list of dicts with keys:
            ``patch_path``, ``label``, ``class_name``.
    """
    if not (1 <= len(fractions) <= 3):
        sys.exit("[ERROR] --split accepts 1, 2, or 3 values (e.g. 0.7 0.2 0.1)")
    total = sum(fractions)
    if total > 1.0 + 1e-6:
        sys.exit(f"[ERROR] --split values sum to {total:.4f} — must be ≤ 1.0")

    n_fracs   = len(fractions)
    f_train   = fractions[0]
    f_val     = fractions[1] if n_fracs >= 2 else 0.0
    f_test    = fractions[2] if n_fracs == 3 else 0.0

    want_val  = n_fracs >= 2
    want_test = n_fracs == 3

    # Discover classes
    class_names = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    )
    if not class_names:
        sys.exit(f"[ERROR] No class sub-folders found in '{data_path}'")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    _IMAGE_EXT   = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    train_items: list[dict] = []
    val_items:   list[dict] | None = [] if want_val  else None
    test_items:  list[dict] | None = [] if want_test else None

    rng = random.Random(seed)
    print(
        f"\nAuto-splitting '{data_path}'  "
        f"train={f_train:.0%}  "
        f"val={'—' if not want_val  else f'{f_val:.0%}'}  "
        f"test={'—' if not want_test else f'{f_test:.0%}'}  "
        f"seed={seed}"
    )

    # Track patch_path sets per split to allow a post-split leak check
    train_paths: set[str] = set()
    val_paths:   set[str] = set()
    test_paths:  set[str] = set()

    for class_name in class_names:
        label     = class_to_idx[class_name]
        class_dir = os.path.join(data_path, class_name)
        files = [
            {
                "patch_path": os.path.join(class_dir, f),
                "label":      label,
                "class_name": class_name,
            }
            for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXT
        ]
        rng.shuffle(files)          # in-place, uses class-local RNG state

        n       = len(files)
        n_train = round(n * f_train)

        # Clamp n_val so n_train + n_val never exceeds n (guards against
        # round() overshooting when fractions sum close to 1.0).
        n_val   = min(round(n * f_val), n - n_train) if want_val  else 0
        # Test gets the true remainder — never negative, never drops images.
        n_test  = (n - n_train - n_val)               if want_test else 0

        # ── Slice into non-overlapping windows ──────────────────
        train_slice = files[:n_train]
        val_slice   = files[n_train : n_train + n_val]
        test_slice  = files[n_train + n_val : n_train + n_val + n_test]

        train_items.extend(train_slice)
        if want_val:
            val_items.extend(val_slice)
        if want_test:
            test_items.extend(test_slice)

        # Accumulate paths for leak check
        train_paths.update(s["patch_path"] for s in train_slice)
        val_paths.update(  s["patch_path"] for s in val_slice)
        test_paths.update( s["patch_path"] for s in test_slice)

        print(
            f"  [{label}] {class_name:>20s}  "
            f"total={n:4d}  train={n_train:4d}  "
            f"val={n_val:4d}  test={n_test:4d}"
        )

    # ── Strict no-leakage assertion ──────────────────────────────
    _assert_no_leakage(train_paths, val_paths, test_paths)

    return train_items, val_items, test_items, class_to_idx


def _assert_no_leakage(
    train_paths: set[str],
    val_paths:   set[str],
    test_paths:  set[str],
) -> None:
    """
    Hard-stop if any image path appears in more than one split.
    Called automatically after every auto-split.
    """
    tv = train_paths & val_paths
    tt = train_paths & test_paths
    vt = val_paths   & test_paths

    leaks = []
    if tv:
        leaks.append(f"  train ∩ val  : {len(tv)} image(s)")
    if tt:
        leaks.append(f"  train ∩ test : {len(tt)} image(s)")
    if vt:
        leaks.append(f"  val   ∩ test : {len(vt)} image(s)")

    if leaks:
        msg = "\n".join(["[FATAL] Data leakage detected between splits:"] + leaks)
        sys.exit(msg)

    print("  ✔ No leakage detected between splits.")


# ===============================================================
# Shared helpers
# ===============================================================
def build_transforms(augment: bool = False) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Builds distinct transform pipelines for training and evaluation.

    Args:
        augment: If True, adds spatial (flips, rotations) and color augmentations
                 to the training set pipeline.

    Returns:
        A tuple of (train_transform, eval_transform).
    """
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if augment:
        train_transform = v2.Compose([
            v2.ToImage(),                              # Convert PIL image to Tensor Image
            v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Resize & Crop
            v2.RandomHorizontalFlip(p=0.5),             # Basic spatial flip
            v2.TrivialAugmentWide(),                    # <--- Automatically applies best augmentations
            v2.ToDtype(torch.float32, scale=True),      # Convert to float and scale pixels to [0, 1]
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ])
    else:
        train_transform = eval_transform

    return train_transform, eval_transform


def load_model(
    results_dir: str,
    clinical_input_dim: int,
    device: torch.device,
    num_classes: int = 21,
) -> nn.Module:
    """Load the best saved model from *results_dir*."""
    model_path = os.path.join(results_dir, "best_model.pt")
    if not os.path.isfile(model_path):
        sys.exit(f"[ERROR] No saved model found at '{model_path}'. Run --mode train first.")

    model = MultimodalResNet18(clinical_input_dim, num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"Loaded model weights from '{model_path}'")
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    results_dir: str,
    split_name: str = "test",
    criterion: nn.Module | None = None,
    class_to_idx: dict | None = None,
) -> dict:
    """
    Run inference on *loader*, compute metrics, and save outputs to *results_dir*.

    Returns a dict of scalar metrics.
    """
    is_binary = class_to_idx is not None and len(class_to_idx) == 2

    model.eval()
    y_true, y_pred, y_prob_all = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for imgs, clinical, labels in tqdm(loader, desc=f"Evaluating [{split_name}]"):
            imgs, clinical = imgs.to(device), clinical.to(device)
            outputs = model(imgs, clinical)

            if criterion is not None:
                total_loss += criterion(outputs, labels.to(device)).item()

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob_all.extend(probs.cpu().numpy())

    y_prob_all = np.array(y_prob_all)   # shape (N, num_classes)

    cm = confusion_matrix(y_true, y_pred)
    avg_loss = total_loss / len(loader) if (criterion is not None and len(loader) > 0) else float("nan")

    avg = "binary" if is_binary else "macro"
    metrics = {
        "Split":      split_name,
        "Loss":       avg_loss,
        "Accuracy":   accuracy_score(y_true, y_pred),
        "Precision":  precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall":     recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1 Score":   f1_score(y_true, y_pred, average=avg, zero_division=0),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
    }

    if is_binary:
        tn, fp, fn, tp = cm.ravel()
        metrics["Specificity"]     = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["True Positives"]  = int(tp)
        metrics["False Positives"] = int(fp)
        metrics["True Negatives"]  = int(tn)
        metrics["False Negatives"] = int(fn)
        metrics["AUC"] = (
            roc_auc_score(y_true, y_prob_all[:, 1])
            if len(set(y_true)) == 2 else float("nan")
        )
    else:
        try:
            metrics["AUC (macro OvR)"] = roc_auc_score(
                y_true, y_prob_all, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["AUC (macro OvR)"] = float("nan")

    # ── Save metrics ──────────────────────────────────────────
    prefix = os.path.join(results_dir, split_name)
    pd.DataFrame([metrics]).to_csv(f"{prefix}_metrics.csv", index=False)
    with open(f"{prefix}_metrics.txt", "w") as f:
        for k, v in metrics.items():
            line = f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n"
            f.write(line)

    # ── Confusion matrix ──────────────────────────────────────
    if class_to_idx:
        tick_labels = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    else:
        tick_labels = list(range(cm.shape[0]))
    plt.figure(figsize=(max(6, 0.5 * len(tick_labels)), max(5, 0.5 * len(tick_labels))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    plt.title(f"Confusion Matrix — {split_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    # ── ROC curve ─────────────────────────────────────────────
    if is_binary and len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob_all[:, 1])
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {split_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{prefix}_roc_curve.png", dpi=150)
        plt.close()

    print(f"\n{'─' * 40}")
    print(f"  Results for split: {split_name}")
    print(f"{'─' * 40}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


# ===============================================================
# Class-map persistence helpers
# ===============================================================
def _save_class_map(class_to_idx: dict, results_dir: str) -> None:
    """Save {class_name: label} mapping to results_dir/class_map.csv."""
    path = os.path.join(results_dir, "class_map.csv")
    rows = sorted(class_to_idx.items(), key=lambda x: x[1])
    pd.DataFrame(rows, columns=["class_name", "label"]).to_csv(path, index=False)
    with open(os.path.join(results_dir, "num_classes.txt"), "w") as f:
        f.write(str(len(class_to_idx)))
    print(f"  Class map saved to '{path}'")


def _load_class_map(results_dir: str) -> dict | None:
    """Load class map saved by training. Returns None if not found."""
    path = os.path.join(results_dir, "class_map.csv")
    if not os.path.isfile(path):
        print(f"[WARNING] No class_map.csv in '{results_dir}' — class labels will be auto-assigned.")
        return None
    df = pd.read_csv(path)
    mapping = dict(zip(df["class_name"], df["label"]))
    print(f"  Loaded class map from '{path}': {mapping}")
    return mapping


def _load_num_classes(results_dir: str, fallback: int = 21) -> int:
    """Read num_classes.txt written during training."""
    path = os.path.join(results_dir, "num_classes.txt")
    if os.path.isfile(path):
        return int(open(path).read().strip())
    print(f"[WARNING] num_classes.txt not found in '{results_dir}', defaulting to {fallback}.")
    return fallback


# ===============================================================
# Mode Implementations
# ===============================================================
def run_train(args) -> None:
    """
    Full training loop with optional inline validation.

    In auto-split mode the class map is saved to disk *before* training
    begins so that a subsequent standalone ``--mode val`` or ``--mode test``
    call can always reload a consistent label mapping, even if training is
    interrupted.

    Val / test split items are **never** seen during training.  The training
    DataLoader only receives ``_split_train`` items; val items are used only
    for the per-epoch loss check; test items are untouched until evaluate steps.
    """
    os.makedirs(args.results_dir, exist_ok=True)

    # Initialize TensorBoard writer
    log_dir = os.path.join(args.results_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] TensorBoard logging initialized at: '{log_dir}'")

    # Generate distinct training and evaluation transforms based on user choices
    train_transform, eval_transform = build_transforms(args.augment)

    # ── Build datasets ─────────────────────────────────────────
    if args._split_train is not None:
        # Auto-split mode — item lists were produced by split_dataset()
        if not args._split_train:
            sys.exit("[ERROR] Auto-split produced an empty training set. "
                     "Check --split fractions and --data path.")
        train_dataset = MultimodalDataset(
            None, train_transform,
            class_to_idx=args._split_class_map,
            items=args._split_train,
        )
        # Save class map immediately so val/test standalone calls can reload it
        _save_class_map(train_dataset.class_to_idx, args.results_dir)

        use_val = args._split_val is not None and len(args._split_val) > 0
        val_dataset_obj = (
            MultimodalDataset(
                None, eval_transform,  # Validation strictly uses non-augmented eval_transform
                class_to_idx=args._split_class_map,
                items=args._split_val,
            )
            if use_val else None
        )
    else:
        # Manual directory mode
        if not args.train_dir:
            sys.exit("[ERROR] --mode train requires --train_dir (or use --data + --split)")
        train_dataset = MultimodalDataset(args.train_dir, train_transform)
        _save_class_map(train_dataset.class_to_idx, args.results_dir)

        use_val = args.val_dir is not None
        val_dataset_obj = (
            MultimodalDataset(args.val_dir, eval_transform,  # Validation strictly uses eval_transform
                              class_to_idx=train_dataset.class_to_idx)
            if use_val else None
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = None
    if use_val:
        val_loader = DataLoader(
            val_dataset_obj, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
    else:
        print("[INFO] No validation data — validation will be skipped during training.")

    clinical_input_dim = len(train_dataset.clinical_cols)
    num_classes        = len(train_dataset.class_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  num_classes: {num_classes}")

    model = MultimodalResNet18(clinical_input_dim, num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Compute class weights to handle imbalance (from training set only)
    all_labels = torch.tensor(
        [s["label"] for s in train_dataset.items], dtype=torch.long
    )
    label_counts = torch.bincount(all_labels)
    class_weights = len(all_labels) / (len(label_counts) * label_counts.float())

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(args.epochs):
        print(f"\n===== START EPOCH {epoch + 1} =====")

        # ── Train ─────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for imgs, clinical, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            imgs         = imgs.to(device)
            clinical     = clinical.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, clinical)
            loss    = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds   = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total   += batch_labels.size(0)

        train_acc      = correct / total
        avg_train_loss = train_loss / len(train_loader)

        # Log training metrics to TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_acc, epoch + 1)

        row = {
            "epoch":      epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc":  train_acc,
        }

        # ── Validate (optional) ───────────────────────────────
        if use_val:
            model.eval()
            val_loss, correct, total = 0.0, 0, 0

            with torch.no_grad():
                for imgs, clinical, batch_labels in val_loader:
                    imgs         = imgs.to(device)
                    clinical     = clinical.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs   = model(imgs, clinical)
                    val_loss += criterion(outputs, batch_labels).item()
                    preds     = outputs.argmax(dim=1)
                    correct  += (preds == batch_labels).sum().item()
                    total    += batch_labels.size(0)

            val_acc      = correct / total
            avg_val_loss = val_loss / len(val_loader)
            row["val_loss"] = avg_val_loss
            row["val_acc"]  = val_acc

            # Log validation metrics to TensorBoard
            writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
            writer.add_scalar("Accuracy/val", val_acc, epoch + 1)

            print(
                f"[Epoch {epoch + 1:3d}] "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(args.results_dir, "best_model.pt"),
                )
                print(f"  ✔ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            print(
                f"[Epoch {epoch + 1:3d}] "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%}"
            )
            torch.save(
                model.state_dict(),
                os.path.join(args.results_dir, "best_model.pt"),
            )

        history.append(row)

    # ── Save training history ──────────────────────────────────
    history_df = pd.DataFrame(history)
    history_df.to_excel(
        os.path.join(args.results_dir, "training_history.xlsx"), index=False
    )

    # ── Learning curves ───────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history_df["epoch"], history_df["train_loss"], label="Train")
    if use_val:
        ax1.plot(history_df["epoch"], history_df["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history_df["epoch"], history_df["train_acc"], label="Train")
    if use_val:
        ax2.plot(history_df["epoch"], history_df["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "learning_curves.png"), dpi=150)
    plt.close()

    # Close TensorBoard writer
    writer.close()
    print("\nTraining complete. Model saved to", args.results_dir)

    # ── Automatic Post-Training Evaluation ──────────────────────
    print("\n" + "=" * 60)
    print("  Running Automatic Post-Training Evaluations")
    print("=" * 60)

    if use_val:
        print("\n--> Auto-evaluating Best Model on Validation Set...")
        run_val(args, eval_transform=eval_transform)

    # Identify if a test partition exists
    use_test = (args._split_test is not None and len(args._split_test) > 0) or (args.test_dir is not None)
    if use_test:
        print("\n--> Auto-evaluating Best Model on Test Set...")
        run_test(args, eval_transform=eval_transform)
    else:
        print("\n[INFO] No test split provided. Skipping automated test evaluation.")


def run_val(args, eval_transform=None) -> None:
    """Evaluate the best saved model on the validation set."""
    os.makedirs(args.results_dir, exist_ok=True)

    # Fallback to base transform if not programmatically provided
    if eval_transform is None:
        _, eval_transform = build_transforms(augment=False)

    class_to_idx = _load_class_map(args.results_dir)

    if args._split_val is not None:
        # Auto-split mode
        if not args._split_val:
            sys.exit(
                "[ERROR] Auto-split produced an empty val set. "
                "Increase the val fraction in --split (use 2 or 3 values)."
            )
        effective_map = args._split_class_map or class_to_idx
        val_dataset   = MultimodalDataset(
            None, eval_transform,
            class_to_idx=effective_map,
            items=args._split_val,
        )
        class_to_idx = val_dataset.class_to_idx
    else:
        if not args.val_dir:
            sys.exit("[ERROR] --mode val requires --val_dir (or use --data + --split)")
        val_dataset  = MultimodalDataset(args.val_dir, eval_transform, class_to_idx=class_to_idx)
        class_to_idx = val_dataset.class_to_idx

    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_classes = _load_num_classes(args.results_dir, fallback=len(class_to_idx) if class_to_idx else 21)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = load_model(args.results_dir, len(val_dataset.clinical_cols), device, num_classes)

    evaluate(model, val_loader, device, args.results_dir, split_name="val", class_to_idx=class_to_idx)


def run_test(args, eval_transform=None) -> None:
    """Evaluate the best saved model on the test set."""
    os.makedirs(args.results_dir, exist_ok=True)

    # Fallback to base transform if not programmatically provided
    if eval_transform is None:
        _, eval_transform = build_transforms(augment=False)

    class_to_idx = _load_class_map(args.results_dir)

    if args._split_test is not None:
        # Auto-split mode
        if not args._split_test:
            sys.exit(
                "[ERROR] Auto-split produced an empty test set. "
                "Increase the test fraction in --split (use 3 values, e.g. 0.5 0.15 0.35)."
            )
        effective_map = args._split_class_map or class_to_idx
        test_dataset  = MultimodalDataset(
            None, eval_transform,
            class_to_idx=effective_map,
            items=args._split_test,
        )
        class_to_idx = test_dataset.class_to_idx
    else:
        if not args.test_dir:
            sys.exit("[ERROR] --mode test requires --test_dir (or use --data + --split)")
        test_dataset = MultimodalDataset(args.test_dir, eval_transform, class_to_idx=class_to_idx)
        class_to_idx = test_dataset.class_to_idx

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_classes = _load_num_classes(args.results_dir, fallback=len(class_to_idx) if class_to_idx else 21)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = load_model(args.results_dir, len(test_dataset.clinical_cols), device, num_classes)

    evaluate(model, test_loader, device, args.results_dir, split_name="test", class_to_idx=class_to_idx)


# ===============================================================
# Argument Parsing
# ===============================================================
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal ResNet18 for image-patch classification (defaults to the UCMerced Land Use dataset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Run mode: 'train', 'val' (validate only), 'test' (test only)",
    )
    parser.add_argument(
        "--train_dir", type=str, default=None,
        help="Path to training data directory (required for --mode train if --data is not used)",
    )
    parser.add_argument(
        "--val_dir", type=str, default=None,
        help="Path to validation data directory",
    )
    parser.add_argument(
        "--test_dir", type=str, default=None,
        help="Path to test data directory (required for --mode test if --data is not used)",
    )
    parser.add_argument(
        "--results_dir", type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save results and model checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # ── Augmentation flag ─────────────────────────────────────
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Activate dataset augmentation (random flips, rotations, jitter) for the training split.",
    )

    # ── Auto-split arguments ──────────────────────────────────
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA_PATH, metavar="DATA_PATH",
        help=(
            "Root data folder (class/images structure). "
            "Used with --split to auto-partition. "
            "Overrides --train_dir / --val_dir / --test_dir. "
            "Set to an empty string ('') to disable auto-split mode and use "
            "--train_dir / --val_dir / --test_dir instead."
        ),
    )
    parser.add_argument(
        "--split", type=float, nargs="+", default=DEFAULT_SPLIT, metavar="FRAC",
        help=(
            "Split fractions summing to ≤ 1.0. "
            "1 value → train only. "
            "2 values → train + val. "
            "3 values → train + val + test. "
            "Used together with --data."
        ),
    )
    parser.add_argument(
        "--split_seed", type=int, default=42, metavar="SEED",
        help="Random seed for reproducible auto-splitting (default: 42).",
    )

    # parse_known_args() so this also runs cleanly inside Jupyter/Kaggle
    # notebook cells, which often inject extra args (e.g. -f kernel.json)
    # that argparse would otherwise choke on.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring unrecognized args (likely from the notebook runtime): {unknown}")

    # Treat an empty string the same as "not provided"
    if args.data == "":
        args.data = None

    return args


# ===============================================================
# Entry Point
# ===============================================================
if __name__ == "__main__":
    args = get_args()

    # ── Auto-split path ────────────────────────────────────────
    if args.data is not None:
        if args.split is None:
            sys.exit("[ERROR] --data requires --split (e.g. --split 0.5 0.15 0.35)")

        train_items, val_items, test_items, class_to_idx = split_dataset(
            args.data, args.split, seed=args.split_seed
        )

        # val_items / test_items are None when that split was not requested
        # (split_dataset returns None, not [], for unused splits).
        args._split_train     = train_items
        args._split_val       = val_items    # None if only 1 fraction given
        args._split_test      = test_items   # None if only 1 or 2 fractions given
        args._split_class_map = class_to_idx

        # Clear dir args so run_* functions use the pre-split item lists
        args.train_dir = None
        args.val_dir   = None
        args.test_dir  = None
    else:
        # Manual directory mode — no pre-split lists
        args._split_train     = None
        args._split_val       = None
        args._split_test      = None
        args._split_class_map = None

    dispatch = {
        "train": run_train,
        "val":   run_val,
        "test":  run_test,
    }
    dispatch[args.mode](args)