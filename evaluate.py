import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# ── Model ──────────────────────────────────────────────────────────────────
class SegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        base = models.resnet50(pretrained=False)
        self.backbone   = nn.Sequential(*list(base.children())[:-2])
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, 1), nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x = self.backbone(x)
        x = self.classifier(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics_per_class(all_preds, all_masks, num_classes=21):
    precision_list = []
    recall_list    = []
    iou_list       = []

    for c in range(num_classes):
        tp = int(((all_preds == c) & (all_masks == c)).sum())
        fp = int(((all_preds == c) & (all_masks != c)).sum())
        fn = int(((all_preds != c) & (all_masks == c)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        precision_list.append(prec)
        recall_list.append(rec)
        iou_list.append(iou)

    pixel_acc = float((all_preds == all_masks).mean()) * 100
    mean_iou  = float(np.mean(iou_list)) * 100

    return {
        'precision':  [p * 100 for p in precision_list],
        'recall':     [r * 100 for r in recall_list],
        'iou':        [i * 100 for i in iou_list],
        'pixel_acc':  pixel_acc,
        'mean_iou':   mean_iou,
        'macro_prec': float(np.mean(precision_list)) * 100,
        'macro_rec':  float(np.mean(recall_list))    * 100,
    }


# ── Evaluate ───────────────────────────────────────────────────────────────
def evaluate(model_path, data_root, batch_size=8, img_size=224):
    from dataset import VOCSegDataset

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {DEVICE}')
    print(f'Model  : {model_path}')

    # Load model
    model = SegmentationModel(num_classes=21).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print('Model loaded ✓')

    # Dataset
    val_set    = VOCSegDataset(root=data_root, split='val', img_size=img_size)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    criterion  = nn.CrossEntropyLoss(ignore_index=255)
    val_loss   = 0.0
    all_preds  = []
    all_masks  = []

    print(f'Running validation on {len(val_set)} images...')
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds        = model(imgs)
            val_loss    += criterion(preds, masks).item()
            pred_labels  = preds.argmax(dim=1).cpu().numpy().flatten()
            mask_labels  = masks.cpu().numpy().flatten()
            all_preds.append(pred_labels)
            all_masks.append(mask_labels)
            if (i + 1) % 20 == 0:
                print(f'  Batch {i+1}/{len(val_loader)}')

    avg_loss     = val_loss / len(val_loader)
    all_preds_np = np.concatenate(all_preds)
    all_masks_np = np.concatenate(all_masks)
    metrics      = compute_metrics_per_class(all_preds_np, all_masks_np)

    print(f'\n── Results ──────────────────────────────')
    print(f'Val Loss    : {avg_loss:.4f}')
    print(f'Pixel Acc   : {metrics["pixel_acc"]:.2f}%')
    print(f'Mean IoU    : {metrics["mean_iou"]:.2f}%')
    print(f'Macro Prec  : {metrics["macro_prec"]:.2f}%')
    print(f'Macro Recall: {metrics["macro_rec"]:.2f}%')

    return avg_loss, metrics


# ── Plot ───────────────────────────────────────────────────────────────────
def save_metrics_plot(avg_loss, metrics, save_path='models/training_metrics.png'):
    BG      = '#0a0a12'
    PANEL   = '#11111c'
    BORDER  = '#1c1c2e'
    ACCENT1 = '#7c6af7'
    ACCENT2 = '#f76a8c'
    ACCENT3 = '#43e8b0'
    ACCENT4 = '#f7c948'
    TEXT    = '#e8e8f2'
    SUBTEXT = '#6060a0'

    classes = VOC_CLASSES
    x       = np.arange(len(classes))
    prec    = metrics['precision']
    rec     = metrics['recall']
    iou     = metrics['iou']

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.06, right=0.97,
                           top=0.88, bottom=0.07)

    # ── 1. Summary cards (top row, spans both cols) ─────────────────────
    ax_cards = fig.add_subplot(gs[0, :])
    ax_cards.set_facecolor(BG)
    ax_cards.axis('off')

    card_data = [
        ('Val Loss',      f'{avg_loss:.4f}',              ACCENT1),
        ('Pixel Acc',     f'{metrics["pixel_acc"]:.2f}%', ACCENT3),
        ('Mean IoU',      f'{metrics["mean_iou"]:.2f}%',  ACCENT4),
        ('Macro Prec',    f'{metrics["macro_prec"]:.2f}%',ACCENT2),
        ('Macro Recall',  f'{metrics["macro_rec"]:.2f}%', '#60d4f7'),
    ]

    for idx, (label, value, color) in enumerate(card_data):
        xpos = 0.05 + idx * 0.19
        rect = FancyBboxPatch((xpos, 0.05), 0.16, 0.85,
                              boxstyle='round,pad=0.02',
                              linewidth=1.5, edgecolor=color,
                              facecolor=PANEL,
                              transform=ax_cards.transAxes, zorder=2)
        ax_cards.add_patch(rect)
        ax_cards.text(xpos + 0.08, 0.62, value,
                      ha='center', va='center', fontsize=17,
                      fontweight='bold', color=color,
                      transform=ax_cards.transAxes, fontfamily='monospace')
        ax_cards.text(xpos + 0.08, 0.25, label,
                      ha='center', va='center', fontsize=9,
                      color=SUBTEXT,
                      transform=ax_cards.transAxes, fontfamily='monospace')

    # ── 2. Per-class Precision ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor(PANEL)
    for spine in ax2.spines.values():
        spine.set_edgecolor(BORDER)
    bars = ax2.bar(x, prec, color=ACCENT1, alpha=0.85, width=0.6)
    # Color bars by value
    for bar, val in zip(bars, prec):
        alpha = 0.4 + 0.6 * (val / 100)
        bar.set_alpha(alpha)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=35, ha='right',
                        fontsize=8, color=SUBTEXT, fontfamily='monospace')
    ax2.set_ylabel('Precision (%)', color=TEXT, fontsize=10)
    ax2.set_title('Per-Class Precision', color=TEXT,
                  fontsize=12, fontweight='bold', pad=10)
    ax2.tick_params(colors=SUBTEXT)
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', color=BORDER, linestyle='--', linewidth=0.7, alpha=0.6)
    ax2.axhline(metrics['macro_prec'], color=ACCENT2, linewidth=1.5,
                linestyle='--', label=f'Macro avg: {metrics["macro_prec"]:.1f}%')
    ax2.legend(facecolor=PANEL, edgecolor=BORDER,
               labelcolor=TEXT, fontsize=9)

    # ── 3. Per-class Recall ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(PANEL)
    for spine in ax3.spines.values():
        spine.set_edgecolor(BORDER)
    ax3.barh(x, rec, color=ACCENT3, alpha=0.8, height=0.6)
    ax3.set_yticks(x)
    ax3.set_yticklabels(classes, fontsize=7.5,
                        color=SUBTEXT, fontfamily='monospace')
    ax3.set_xlabel('Recall (%)', color=TEXT, fontsize=10)
    ax3.set_title('Per-Class Recall', color=TEXT,
                  fontsize=12, fontweight='bold', pad=10)
    ax3.tick_params(colors=SUBTEXT)
    ax3.set_xlim(0, 110)
    ax3.grid(axis='x', color=BORDER, linestyle='--', linewidth=0.7, alpha=0.6)
    ax3.axvline(metrics['macro_rec'], color=ACCENT4, linewidth=1.5,
                linestyle='--', label=f'Macro: {metrics["macro_rec"]:.1f}%')
    ax3.legend(facecolor=PANEL, edgecolor=BORDER,
               labelcolor=TEXT, fontsize=9)

    # ── 4. Per-class IoU ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor(PANEL)
    for spine in ax4.spines.values():
        spine.set_edgecolor(BORDER)
    colors_iou = [ACCENT4 if v >= metrics['mean_iou'] else ACCENT1 for v in iou]
    ax4.bar(x, iou, color=colors_iou, alpha=0.85, width=0.6)
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes, rotation=35, ha='right',
                        fontsize=8, color=SUBTEXT, fontfamily='monospace')
    ax4.set_ylabel('IoU (%)', color=TEXT, fontsize=10)
    ax4.set_title('Per-Class IoU  (gold = above mean)',
                  color=TEXT, fontsize=12, fontweight='bold', pad=10)
    ax4.tick_params(colors=SUBTEXT)
    ax4.set_ylim(0, 110)
    ax4.grid(axis='y', color=BORDER, linestyle='--', linewidth=0.7, alpha=0.6)
    ax4.axhline(metrics['mean_iou'], color=ACCENT2, linewidth=1.5,
                linestyle='--', label=f'mIoU: {metrics["mean_iou"]:.1f}%')
    ax4.legend(facecolor=PANEL, edgecolor=BORDER,
               labelcolor=TEXT, fontsize=9)

    # ── Title ───────────────────────────────────────────────────────────
    fig.text(0.5, 0.95,
             'Semantic Segmentation — Evaluation Metrics',
             ha='center', fontsize=18, fontweight='bold',
             color=TEXT, fontfamily='monospace')
    fig.text(0.5, 0.915,
             'ResNet-50  ·  Pascal VOC 2012  ·  21 Classes',
             ha='center', fontsize=10, color=SUBTEXT, fontfamily='monospace')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'\n✓ Metrics image saved → {save_path}')


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/segmentation_model.pth')
    parser.add_argument('--data',  default='./data')
    parser.add_argument('--out',   default='models/training_metrics.png')
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    avg_loss, metrics = evaluate(
        model_path=args.model,
        data_root=args.data,
        batch_size=args.batch,
    )
    save_metrics_plot(avg_loss, metrics, save_path=args.out)