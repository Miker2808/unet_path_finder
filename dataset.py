import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class MassachusettsRoadsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, crop_size=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.crop_size = crop_size
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".tiff",".tif"))
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask >= 100.0] = 1.0

        if self.crop_size is not None:
            h, w = image.shape[:2]
            crop_h, crop_w = self.crop_size
            
            # Random top-left corner
            y = np.random.randint(0, h - crop_h + 1)
            x = np.random.randint(0, w - crop_w + 1)
            
            image = image[y:y+crop_h, x:x+crop_w]
            mask = mask[y:y+crop_h, x:x+crop_w]

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask