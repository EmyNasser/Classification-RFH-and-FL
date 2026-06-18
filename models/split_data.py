from sklearn.model_selection import train_test_split
import os
import shutil

src = "/kaggle/input/datasets/eman12345nasser/ucmerced-landuse/Images"
output = "/kaggle/working/data"

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output, split), exist_ok=True)

for cls in os.listdir(src):

    cls_path = os.path.join(src, cls)

    if not os.path.isdir(cls_path):
        continue

    imgs = [
        os.path.join(cls_path, img)
        for img in os.listdir(cls_path)
        if img.lower().endswith(
            (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        )
    ]

    train_imgs, temp_imgs = train_test_split(
        imgs,
        test_size=0.30,
        random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.50,
        random_state=42
    )

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split_name, split_imgs in splits.items():

        save_dir = os.path.join(
            output,
            split_name,
            cls
        )

        os.makedirs(save_dir, exist_ok=True)

        for img_path in split_imgs:
            shutil.copy2(img_path, save_dir)

print("Done.")