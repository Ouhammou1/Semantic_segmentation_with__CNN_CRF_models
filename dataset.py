# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class VOCSegDataset(Dataset):
    def __init__(self, root, split='train', img_size=224):
        self.img_size = img_size

        base = os.path.join(root, 'VOC2012_train_val/VOC2012_train_val')
        self.img_dir  = os.path.join(base, 'JPEGImages')
        self.mask_dir = os.path.join(base, 'SegmentationClass')

        if split == 'train':
            split_file = os.path.join(base, 'ImageSets/Segmentation/train.txt')
        else:
            split_file = os.path.join(base, 'ImageSets/Segmentation/val.txt')

        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

        print('[' + split + '] Found ' + str(len(self.ids)) + ' images')

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path  = os.path.join(self.img_dir,  img_id + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_id + '.png')

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path)

        image = self.img_transform(image)

        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        mask[mask == 255] = 0

        return image, torch.tensor(mask, dtype=torch.long)


if __name__ == '__main__':
    train_set = VOCSegDataset(root='./data', split='train')
    val_set   = VOCSegDataset(root='./data', split='val')

    loader = DataLoader(train_set, batch_size=4, shuffle=True)
    imgs, masks = next(iter(loader))
    print('Images : ' + str(imgs.shape))
    print('Masks  : ' + str(masks.shape))
    print('Classes: ' + str(masks.unique().tolist()))