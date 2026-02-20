# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import VOCSegDataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ── Model ──────────────────────────────────────────────────────────────────
class SegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super(SegmentationModel, self).__init__()
        base = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x = self.backbone(x)
        x = self.classifier(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)


# ── Train ──────────────────────────────────────────────────────────────────
def train():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 20
    BATCH  = 8
    LR     = 1e-4

    print('Using device: ' + str(DEVICE))

    train_set = VOCSegDataset(root='./data', split='train', img_size=224)
    val_set   = VOCSegDataset(root='./data', split='val',   img_size=224)

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False, num_workers=2)

    model     = SegmentationModel(num_classes=21).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_val_loss = float('inf')
    os.makedirs('models', exist_ok=True)

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss  = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 20 == 0:
                print('  Step [' + str(i+1) + '/' + str(len(train_loader)) + '] Loss: ' + str(round(loss.item(), 4)))

        # Validation
        model.eval()
        val_loss = 0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()
                pred_labels = preds.argmax(dim=1)
                correct += (pred_labels == masks).sum().item()
                total   += masks.numel()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        pixel_acc = correct / total * 100

        print('Epoch [' + str(epoch+1).zfill(2) + '/' + str(EPOCHS) + ']' +
              ' Train Loss: ' + str(round(avg_train, 4)) +
              ' | Val Loss: ' + str(round(avg_val, 4)) +
              ' | Pixel Acc: ' + str(round(pixel_acc, 2)) + '%')

        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'models/segmentation_model.pth')
            print('  >>> Best model saved (val loss: ' + str(round(avg_val, 4)) + ')')

    print('Training complete!')


if __name__ == '__main__':
    train()