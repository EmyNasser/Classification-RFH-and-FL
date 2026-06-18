from sklearn.model_selection import train_test_split
import os
import shutil

src = "/kaggle/input/datasets/eman12345nasser/ucmerced-landuse"

output = "/kaggle/working/data"

for split in ["train","val","test"]:
    os.makedirs(os.path.join(output,split),exist_ok=True)

for cls in os.listdir(src):

    cls_path = os.path.join(src,cls)

    if not os.path.isdir(cls_path):
        continue

    imgs = [os.path.join(cls_path,x) for x in os.listdir(cls_path)]

    train,valtest = train_test_split(
        imgs,
        test_size=0.3,
        random_state=42
    )

    val,test = train_test_split(
        valtest,
        test_size=0.5,
        random_state=42
    )

    for split,data in zip(
        ["train","val","test"],
        [train,val,test]
    ):

        save_dir = os.path.join(output,split,cls)

        os.makedirs(save_dir,exist_ok=True)

        for img in data:
            shutil.copy(img,save_dir)