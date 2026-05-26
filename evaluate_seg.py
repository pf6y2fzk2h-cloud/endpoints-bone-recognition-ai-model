#!/usr/bin/env python3.12
"""
Evaluate Stage 1 — FingerSegUNet — with IoU per finger class.
Mirrors the evaluation metric reported in Chen (2023).

Run:
    python3.12 evaluate_seg.py
"""

import sys; sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image
import numpy as np
import cv2
import os
import json

CONFIG = {
    'annotations_file': '../labels.json',
    'images_dir':       '../training_images',
    'seg_model_path':   'trained_model/best_seg_model.pth',
    'image_size':       256,
    'seg_threshold':    0.2,
}

FINGER_LABELS = ['L2', 'L4', 'R2', 'R4']

# =============================================================================
# MODEL (must match train_finger_seg.py exactly)
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class FingerSegUNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        bb = tv_models.resnet18(weights=None)
        self.enc0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            bb.bn1, nn.ReLU(inplace=True),
        )
        self.pool = bb.maxpool
        self.enc1 = bb.layer1; self.enc2 = bb.layer2
        self.enc3 = bb.layer3; self.enc4 = bb.layer4
        self.up4 = Up(512, 256, 256); self.up3 = Up(256, 128, 128)
        self.up2 = Up(128,  64,  64); self.up1 = Up(64,   64,  32)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2), ConvBlock(32, 32))
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e0 = self.enc0(x); e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1); e3 = self.enc3(e2); b = self.enc4(e3)
        d = self.up4(b, e3); d = self.up3(d, e2); d = self.up2(d, e1)
        d = self.up1(d, e0); d = self.up0(d)
        return self.head(d)

# =============================================================================
# UTILITIES
# =============================================================================

def apply_clahe(arr):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(arr)


def make_gt_masks(polygon_groups, orig_w, orig_h, size):
    """Rasterise polygon annotations into binary masks at model resolution."""
    masks = np.zeros((4, size, size), dtype=np.float32)
    for pg in polygon_groups:
        label = pg.get('label', '')
        if label not in FINGER_LABELS:
            continue
        ch = FINGER_LABELS.index(label)
        for poly in pg.get('polygons', []):
            if poly.get('hole', False):
                continue
            pts = np.array([
                [int(p['x'] / orig_w * size), int(p['y'] / orig_h * size)]
                for p in poly['points']
            ], dtype=np.int32)
            cv2.fillPoly(masks[ch], [pts], 1.0)
    return masks


def iou_per_class(pred_binary, gt_binary):
    """pred_binary, gt_binary: (4, H, W) numpy arrays with values 0/1."""
    ious = []
    for ch in range(4):
        p = pred_binary[ch].astype(bool)
        g = gt_binary[ch].astype(bool)
        intersection = (p & g).sum()
        union        = (p | g).sum()
        ious.append(float(intersection) / float(union + 1e-6))
    return ious  # list of 4 IoU values

# =============================================================================
# MAIN
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(CONFIG['seg_model_path']):
        print(f"ERROR: {CONFIG['seg_model_path']} not found."); return

    ck = torch.load(CONFIG['seg_model_path'], map_location=device)
    model = FingerSegUNet(4).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print("Segmentation model loaded.\n")

    with open(CONFIG['annotations_file']) as f:
        all_data = json.load(f)

    # Only use images that have all 4 finger polygons
    candidates = []
    for img_data in all_data:
        path = os.path.join(CONFIG['images_dir'], img_data['name'])
        if not os.path.exists(path):
            continue
        poly_labels = {pg['label'] for pg in img_data.get('polygonGroups', [])}
        if all(fl in poly_labels for fl in FINGER_LABELS):
            candidates.append(img_data)

    print(f"Images with all 4 finger annotations: {len(candidates)}\n")
    if not candidates:
        print("No fully annotated images found."); return

    all_ious = {fl: [] for fl in FINGER_LABELS}
    size = CONFIG['image_size']
    thr  = CONFIG['seg_threshold']

    for img_data in candidates:
        arr = np.array(Image.open(
            os.path.join(CONFIG['images_dir'], img_data['name'])).convert('L'),
            dtype=np.uint8)
        orig_h, orig_w = arr.shape

        # Preprocess
        enhanced = apply_clahe(arr)
        resized  = cv2.resize(enhanced, (size, size))
        t = torch.from_numpy(resized.astype(np.float32) / 255.0)
        t = t.unsqueeze(0).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            probs = torch.sigmoid(model(t))[0].cpu().numpy()  # (4, H, W)

        pred_binary = (probs > thr).astype(np.uint8)

        # Ground truth masks
        gt_masks = make_gt_masks(
            img_data.get('polygonGroups', []), orig_w, orig_h, size)
        gt_binary = (gt_masks > 0.5).astype(np.uint8)

        # IoU per finger
        ious = iou_per_class(pred_binary, gt_binary)
        for ch, fl in enumerate(FINGER_LABELS):
            all_ious[fl].append(ious[ch])

    # ── Results ──────────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"SEGMENTATION IoU  ({len(candidates)} images)")
    print("=" * 50)
    mean_ious = []
    for fl in FINGER_LABELS:
        vals = all_ious[fl]
        m = float(np.mean(vals)) * 100
        mean_ious.append(m)
        print(f"  {fl}:  {m:.2f}%  (std {np.std(vals)*100:.2f}%)")

    miou = float(np.mean(mean_ious))
    print(f"\n  mIoU (mean over 4 fingers): {miou:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
