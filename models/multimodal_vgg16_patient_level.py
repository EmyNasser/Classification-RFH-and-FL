"""
Multimodal VGG16 Classifier for Reactive Follicular Hyperplasia and Follicular Lymphoma
-----------------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Description:
    This script implements a multimodal deep learning pipeline that integrates
    histopathological image patches, clinicopathologic, and nuclear morphometric features
    to classify lymphoid lesions into Reactive Follicular Hyperplasia (RFH) and Follicular Lymphoma (FL).

    The model uses VGG16 for feature extraction from image patches, combines it with
    a clinical/nuclear feature MLP, and evaluates performance at both patch-level and patient-level
    using ROC AUC, calibration curves, confusion matrices, and bootstrapped confidence intervals.

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
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------------
# Dataset Definition
class MultimodalDataset(Dataset):

    def __init__(self, image_dir, transform=None):

        self.transform = transform
        self.items = []

        classes = sorted([
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        ])

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(classes)
        }

        for cls in classes:

            cls_dir = os.path.join(image_dir, cls)

            for root, _, files in os.walk(cls_dir):

                for file in files:

                    if file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                    ):

                        self.items.append({
                            "patch_path": os.path.join(root, file),
                            "label": self.class_to_idx[cls]
                        })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        sample = self.items[idx]

        image = Image.open(
            sample["patch_path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        clinical = torch.zeros(
            1,
            dtype=torch.float32
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return image, clinical, label
# ---------------------------------------------------------------------------
# Multimodal VGG16 Model Definition
# ---------------------------------------------------------------------------
class MultimodalVGG16(nn.Module):
    def __init__(self, clinical_input_dim, num_classes=2):
        super().__init__()
        backbone = vgg16(weights=VGG16_Weights.DEFAULT)
        backbone.classifier = nn.Identity()  # Remove original classifier
        self.backbone = backbone.features
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten()
        self.backbone_feature_dim = 512 * 7 * 7  # After VGG16 feature extractor

        self.dropout = nn.Dropout(0.5)

        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_feature_dim + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, clinical_data):
        x = self.backbone(image)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)

        clinical_features = self.clinical_net(clinical_data)
        combined = torch.cat((x, clinical_features), dim=1)

        return self.classifier(combined)




# ---------------------------------------------------------------------------
# Main Training and Evaluation Routine
# ---------------------------------------------------------------------------
def main():
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    results_dir = "/kaggle/working/results"
    os.makedirs(results_dir, exist_ok=True)
    from torch.utils.data import random_split
    # ------------------------------
    # Data
    # ------------------------------
    
    data_dir = "/kaggle/input/datasets/eman12345nasser/ucmerced-landuse/Images"
    full_dataset = MultimodalDataset(
    data_dir,
    transform
    )   

    print("Total Images:", len(full_dataset))
    print("Classes:", len(full_dataset.class_to_idx))
    n = len(full_dataset)

    train_size = int(0.5 * n)
    #val_size   = int(0.15 * n)
    test_size  = n - train_size 
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    clinical_input_dim = 1

    num_classes = len(full_dataset.class_to_idx)    
    model = MultimodalVGG16(clinical_input_dim,
        num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training
    model.train()
    for images, clinical, labels in tqdm(train_loader, desc="Training"):
        images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images, clinical), labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, clinical, labels in tqdm(test_loader, desc="Testing"):
            images, clinical = images.to(device), clinical.to(device)
            outputs = model(images, clinical)
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("Results and plots saved.")


if __name__ == "__main__":
    main()
