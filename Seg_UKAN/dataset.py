import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

        # Filter img_ids to include only those that have corresponding masks
        self.img_ids = [
            img_id for img_id in img_ids
            if os.path.exists(os.path.join(self.mask_dir, img_id + self.mask_ext))
        ]

        print(f"Filtered {len(self.img_ids)} valid image-mask pairs out of {len(img_ids)} images.")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found at path: {mask_path}")

        mask = mask[..., None]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        if mask.max() < 1:
            mask[mask > 0] = 1.0

        return img, mask, {'img_id': img_id}


