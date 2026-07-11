"""
Multimodal AlexNet Classifier for Reactive Follicular Hyperplasia and Follicular Lymphoma
---------------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Description:
    This script implements a multimodal deep learning pipeline that integrates
    histopathological image patches, clinicopathologic and nuclear morphometric features to classify
    lymphoid lesions into Reactive Follicular Hyperplasia (RFH) and Follicular Lymphoma (FL).

    The model is based on an AlexNet convolutional backbone for image embeddings,
    combined with a fully‑connected network for clinical and nuclear morphometric data.
    The network is trained and validated on labeled patch‑level data, and evaluated on
    an external test set using standard classification metrics and explainable outputs.

Dependencies:
    torch>=2.1.0
    torchvision>=0.16.0
    pandas>=2.0.0
    numpy>=1.24.0
    matplotlib>=3.8.0
    seaborn>=0.13.0
    scikit‑learn>=1.3.0
    pillow>=10.0.0
    tqdm>=4.66.0
    openpyxl>=3.1.0
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
    roc_curve,
    classification_report,
    top_k_accuracy_score
)
from tqdm import tqdm
import json
import random
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(42)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from sklearn.model_selection import train_test_split
# ===============================================================
# Dataset Definition
# ===============================================================
class MultimodalDataset(Dataset):

    def __init__(self, image_dir, transform=None):

        self.transform = transform
        self.items = []

        classes = sorted([
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        ])

        self.class_to_idx = {
            cls:i for i, cls in enumerate(classes)
        }
        self.idx_to_class = {
            i: cls
            for cls, i in self.class_to_idx.items()
        }
        for cls in classes:

            cls_dir = os.path.join(image_dir, cls)

            for file in os.listdir(cls_dir):

                if file.lower().endswith(
                    (".png",".jpg",".jpeg",".tif",".tiff")
                ):

                    self.items.append({
                        "patch_path": os.path.join(cls_dir, file),
                        "label": self.class_to_idx[cls],
                        "class_name": cls
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

        # dummy clinical feature
        clinical = torch.zeros(
            1,
            dtype=torch.float32
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return image, clinical, label

# ===============================================================
# Model Definition (AlexNet + Clinical Data)
# ===============================================================
class MultimodalAlexNet(nn.Module):
    def __init__(self, clinical_input_dim, num_classes=21):
        super().__init__()
        backbone = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT
        )
        # remove last classifier layer
        backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])
        self.backbone = backbone
        self.feature_dim = backbone.classifier[-1].in_features if hasattr(backbone.classifier[-1], "in_features") else 4096

        self.dropout = nn.Dropout(0.5)
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, clinical_data):
        x = self.backbone(image)
        x = self.dropout(x)
        clinical_features = self.clinical_net(clinical_data)
        combined = torch.cat((x, clinical_features), dim=1)
        return self.classifier(combined)
        
    # ===============================================================
# Training + Evaluation Pipeline
# ===============================================================
def main():
    # ------------------------------
    # Paths
    # ------------------------------
    checkpoint_path = os.path.join(
            results_dir,
            "last_model.pt"
        )

    if os.path.exists(checkpoint_path):

        print("Loading checkpoint...")

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device
            )
        )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])
    results_dir = "/kaggle/working/results"
    writer = SummaryWriter(
        log_dir=os.path.join(results_dir, "tensorboard")
    )
    os.makedirs(results_dir, exist_ok=True)
    # ------------------------------
    # Data
    # ------------------------------
    
    data_dir = "/kaggle/input/datasets/eman12345nasser/ucmerced-landuse/Images"
    full_dataset = MultimodalDataset(
        data_dir,
        train_transform
    )
    
    print("Total images =", len(full_dataset))
    print("Classes =", full_dataset.class_to_idx)
    pd.DataFrame({
        "Class": list(full_dataset.class_to_idx.keys()),
        "Index": list(full_dataset.class_to_idx.values())
    }).to_csv(
    os.path.join(results_dir, "class_map.csv"),
    index=False
    ) 
    import json

    with open(
        os.path.join(
            results_dir,
            "class_map.json"
        ),
        "w"
    ) as f:

        json.dump(
            full_dataset.class_to_idx,
            f,
            indent=4
        )   
    n = len(full_dataset)

    indices = np.arange(len(full_dataset))

    labels = np.array([
        item["label"]
        for item in full_dataset.items
    ])

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=42
    )

    temp_labels = labels[temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=42
    )

    from torch.utils.data import Subset

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)
    test_dataset  = Subset(full_dataset, test_idx)
    train_paths = set(
        full_dataset.items[i]["patch_path"]
        for i in train_idx
    )

    val_paths = set(
        full_dataset.items[i]["patch_path"]
        for i in val_idx
    )

    test_paths = set(
        full_dataset.items[i]["patch_path"]
        for i in test_idx
    )

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    print("✓ No Data Leakage")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    clinical_input_dim = 1
    num_classes = len(full_dataset.class_to_idx)
    
    model = MultimodalAlexNet(
        clinical_input_dim,
        num_classes=num_classes
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    train_labels = [
    full_dataset.items[i]["label"]
    for i in train_dataset.indices
    ]

    labels = torch.tensor(
        train_labels,
        dtype=torch.long
    )
    label_counts = torch.bincount(labels)
    class_weights = len(labels) / (len(label_counts) * label_counts.float())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )
    # ------------------------------
    # Training
    # ------------------------------
    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available()
    )
    best_val_loss = float("inf")
    patience = 10
    counter = 0
    history = []

    for epoch in range(100):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for imgs, clinical, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):

                outputs = model(imgs, clinical)

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
       
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, clinical, labels in val_loader:
                imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
                with torch.cuda.amp.autocast(
                    enabled=torch.cuda.is_available()
                ):

                    outputs = model(imgs, clinical)

                    loss = criterion(outputs, labels)

                

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar(
            "Loss/train",
            avg_train_loss,
            epoch
        )

        writer.add_scalar(
            "Accuracy/train",
            train_acc,
            epoch
        )

        writer.add_scalar(
            "Loss/validation",
            avg_val_loss,
            epoch
        )

        writer.add_scalar(
            "Accuracy/validation",
            val_acc,
            epoch
        )
        scheduler.step(avg_val_loss)
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })
        history[-1]["lr"] = optimizer.param_groups[0]["lr"]

        history[-1]["best_loss"] = best_val_loss
        print(f"[{epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss
            counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(results_dir,"best_model.pt")
            )
           
        else:

            counter += 1

            if counter >= patience:

                print("Early stopping")

                break
            pd.DataFrame(history).to_excel(os.path.join(results_dir, "training_history.xlsx"), index=False)

        torch.save(
                model.state_dict(),
                os.path.join(
                    results_dir,
                    "last_model.pt"
                )
        )
        history_df = pd.DataFrame(history)
        history_df = pd.DataFrame(history)

        plt.figure(figsize=(8,5))

        plt.plot(

            history_df["epoch"],

            history_df["lr"]

        )

        plt.xlabel("Epoch")

        plt.ylabel("Learning Rate")

        plt.grid(True)

        plt.savefig(

            os.path.join(

                results_dir,

                "learning_rate.png"

            )

        )

        plt.close()
        plt.figure(figsize=(8,5))
        plt.plot(history_df["epoch"], history_df["train_loss"], label="Train")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(
            os.path.join(results_dir,"loss_curve.png")
        )
        plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(history_df["epoch"], history_df["train_acc"], label="Train")
        plt.plot(history_df["epoch"], history_df["val_acc"], label="Validation")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(
            os.path.join(results_dir,"accuracy_curve.png")
        )
        plt.close()
    best_epoch = history_df.loc[

        history_df["val_loss"].idxmin(),

        "epoch"

    ]

    print(

        f"Best Epoch = {best_epoch}"

    )
    # ------------------------------
    # Evaluation
    # ------------------------------
    model.load_state_dict(
        torch.load(
            os.path.join(
                results_dir,
                "best_model.pt"
            ),
            map_location=device
        )
    )
    print("\n🔍 Evaluating on test set...")
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for imgs, clinical, labels in tqdm(test_loader):
            imgs, clinical = imgs.to(device), clinical.to(device)
            outputs = model(imgs, clinical)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(full_dataset.class_to_idx.keys()),
        digits=4
    )

    with open(
        os.path.join(
            results_dir,
            "classification_report.txt"
        ),
        "w"
    ) as f:

        f.write(report)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "Loss (Val Last)": history[-1]["val_loss"] if history else float("nan"),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "Recall (Sensitivity)": recall_score(y_true, y_pred,  average="macro",zero_division=0),
        "F1 Score": f1_score(y_true, y_pred,  average="macro",zero_division=0),
        "Specificity": specificity,
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        #"AUC": roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan"),
        "True Positives": int(tp),
        "False Positives": int(fp),
        "True Negatives": int(tn),
        "False Negatives": int(fn)
    }
    all_probs = []

    with torch.no_grad():

        for imgs, clinical, labels in test_loader:

            imgs = imgs.to(device)
            clinical = clinical.to(device)

            outputs = model(imgs, clinical)

            probs = torch.softmax(
                outputs,
                dim=1
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    metrics["Top5 Accuracy"] = top_k_accuracy_score(
        y_true,
        np.array(all_probs),
        k=5
    )
    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(full_dataset.class_to_idx.keys()),
        yticklabels=list(full_dataset.class_to_idx.keys())
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(7,6))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "roc_curve.png"))
        plt.close()
    writer.close()
    experiment = {

        "Model":"AlexNet",

        "Epochs":100,

        "Batch Size":64,

        "Optimizer":"AdamW",

        "Learning Rate":1e-4,

        "Weight Decay":1e-4,

        "Classes":len(full_dataset.class_to_idx),

        "Dataset Size":len(full_dataset)

    }

    with open(
        os.path.join(
            results_dir,
            "experiment.json"
        ),
        "w"
    ) as f:

        json.dump(
            experiment,
            f,
            indent=4
        )
    predictions = []

    with torch.no_grad():

        for imgs, clinical, labels in test_loader:

            imgs = imgs.to(device)

            clinical = clinical.to(device)

            outputs = model(imgs, clinical)

            probs = torch.softmax(outputs,1)

            conf,preds = probs.max(1)

            for gt,p,c in zip(
                labels,
                preds.cpu(),
                conf.cpu()
            ):

                predictions.append({

                    "GroundTruth":int(gt),

                    "Prediction":int(p),

                    "Confidence":float(c)

                })

    pd.DataFrame(
        predictions
    ).to_csv(

        os.path.join(
            results_dir,
            "predictions.csv"
        ),

        index=False
    )
# ===============================================================
# Entry Point
# ===============================================================

######Inference
def predict_image(

    model,

    image_path,

    class_names,

    device

):

    transform = transforms.Compose([

        transforms.Resize((224,224)),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.485,0.456,0.406],

            [0.229,0.224,0.225]

        )

    ])

    image = Image.open(

        image_path

    ).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    clinical = torch.zeros(

        (1,1),

        device=device

    )

    model.eval()

    with torch.no_grad():

        output = model(

            image,

            clinical

        )

        prob = torch.softmax(

            output,

            dim=1

        )

        confidence,pred = prob.max(1)

    print(

        "Prediction:",

        class_names[pred.item()]

    )

    print(

        "Confidence:",

        confidence.item()

    )
if __name__ == "__main__":
    main()
