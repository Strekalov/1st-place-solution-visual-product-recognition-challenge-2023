import torch.utils.data as data
from turbojpeg import TurboJPEG
import torch
import cv2
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    return A.Compose(
        [
            A.Resize(
                always_apply=False,
                p=1.0,
                height=256+12,
                width=256+12,
                interpolation=cv2.INTER_AREA,
            ),
            
            A.CenterCrop(height=256,
                         width=256, 
                         ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class ThreshDataset(data.Dataset):
    def __init__(self, root):
        self.root = Path(root)

        self.images_path = self.root.rglob("*.jpg")
        self.images_path = [str(path) for path in self.root.rglob("*.jpg") if not Path.exists(Path(f"{path.parent}/{path.stem}.pt"))]
        # self.images_path = [str(path) for path in self.root.rglob("*.jpg")]
        self.turbo_jpeg = TurboJPEG()
        self.transforms = get_transforms()

    def __getitem__(self, index):
        cv2.setNumThreads(120)

        full_imname = self.images_path[index]

        try:
            with open(full_imname, mode="rb") as image_file:
                img = self.turbo_jpeg.decode(image_file.read())
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as ex:
            print(full_imname)
            os.remove(full_imname)
            return torch.zeros(3, 224, 224), 0

        img = self.transforms(image=img)["image"]
        
        return img, full_imname

    def __len__(self):
        return len(self.images_path)
    
    