#!/usr/bin/env python3.12
"""
Stage 1 — Dedicated U-Net Finger Segmentation
===============================================
Trains a full U-Net (all layers free, no frozen backbone) to predict
binary masks for the 4 fingers: L2, L4, R2, R4.

Why a separate model instead of the joint multitask head?
  - No frozen backbone constraints: every layer adapts to fingers
  - U-Net skip connections give sharp, spatially precise masks
  - Segmentation is the ONLY task, so the whole model focuses on it
  - 100 images is enough for a focused U-Net on a specific domain

Output: trained_model/best_seg_model.pth
"""

import sys; sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import json
import numpy as np
import cv2
import os
from PIL import Image
import time

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    'annotations_file': '../labels.json',
    'images_dir':       '../training_images',
    'image_size':       256,    # 256 needed to distinguish adjacent fingers reliably
    'num_fingers':      4,
    'train_split':      0.85,
    'batch_size':       4,
    'lr':               0.001,
    'epochs':           25,     # U-Net with skip connections converges fast
    'pos_weight':       15.0,   # finger pixels are ~5-10% of image
}

FINGER_LABELS = ['L2', 'L4', 'R2', 'R4']

# =============================================================================
# U-NET BUILDING BLOCKS
# =============================================================================

class ConvBlock(nn.Module):
    """Two 3×3 conv layers with BatchNorm + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    """MaxPool then ConvBlock — halves spatial size."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2d to upsample, concat with skip, then ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:  # guard for odd sizes
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# =============================================================================
# MODEL
# =============================================================================

class FingerSegUNet(nn.Module):
    """
    U-Net with pretrained ResNet18 encoder (ImageNet weights, grayscale-adapted).
    The encoder provides rich pretrained features; the decoder is trained from scratch.
    Using standard torchvision ResNet18 — no extra packages required.

    Input:  (B, 1, 256, 256)
    Output: (B, 4, 256, 256) raw logits
    """
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        try:
            bb = tv_models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:                          # older torchvision
            bb = tv_models.resnet18(pretrained=pretrained)

        # Adapt first conv for grayscale: average the 3-channel pretrained weights
        self.enc0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            bb.bn1,
            nn.ReLU(inplace=True),
        )
        if pretrained:
            with torch.no_grad():
                self.enc0[0].weight.data = bb.conv1.weight.data.mean(dim=1, keepdim=True)

        self.pool = bb.maxpool   # stride-2 pool → H/4
        self.enc1 = bb.layer1   # (B,  64, H/4,  W/4)
        self.enc2 = bb.layer2   # (B, 128, H/8,  W/8)
        self.enc3 = bb.layer3   # (B, 256, H/16, W/16)
        self.enc4 = bb.layer4   # (B, 512, H/32, W/32)

        # Decoder — always trained from scratch
        self.up4 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up2 = Up(128,  64,  64)
        self.up1 = Up(64,   64,  32)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            ConvBlock(32, 32),
        )
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e0 = self.enc0(x)              # (B,  64, H/2,  W/2)
        e1 = self.enc1(self.pool(e0))  # (B,  64, H/4,  W/4)
        e2 = self.enc2(e1)             # (B, 128, H/8,  W/8)
        e3 = self.enc3(e2)             # (B, 256, H/16, W/16)
        b  = self.enc4(e3)             # (B, 512, H/32, W/32)
        d  = self.up4(b,  e3)
        d  = self.up3(d,  e2)
        d  = self.up2(d,  e1)
        d  = self.up1(d,  e0)
        d  = self.up0(d)
        return self.head(d)

# =============================================================================
# DATASET
# =============================================================================

def apply_clahe(arr):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr)


class FingerSegDataset(Dataset):
    def __init__(self, annotations_file, images_dir, image_size=512, augment=False):
        with open(annotations_file) as f:
            data = json.load(f)
        self.images_dir = images_dir
        self.image_size = image_size
        self.augment    = augment

        self.samples = []
        for img in data:
            poly_labels = {pg['label'] for pg in img.get('polygonGroups', [])}
            has_all = all(fl in poly_labels for fl in FINGER_LABELS)
            path    = os.path.join(images_dir, img['name'])
            if has_all and os.path.exists(path):
                self.samples.append(img)

        print(f"  FingerSegDataset: {len(self.samples)} images  augment={augment}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_data = self.samples[idx]
        arr      = np.array(Image.open(
            os.path.join(self.images_dir, img_data['name'])).convert('L'),
            dtype=np.uint8)
        orig_h, orig_w = arr.shape

        arr   = apply_clahe(arr)
        masks = self._make_masks(img_data['polygonGroups'], orig_w, orig_h)

        if self.augment:
            if np.random.rand() < 0.5:
                arr, masks = self._flip(arr, masks)
            if np.random.rand() < 0.4:
                arr, masks = self._rotate(arr, masks)
            if np.random.rand() < 0.4:
                arr, masks = self._elastic_deform(arr, masks)
            if np.random.rand() < 0.3:
                arr, masks = self._perspective(arr, masks)
            if np.random.rand() < 0.5:
                arr = self._brightness(arr)

        s   = self.image_size
        img = cv2.resize(arr, (s, s), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

        # Resize masks to model output size (same as input here)
        masks_r = np.stack([
            cv2.resize(masks[c], (s, s), interpolation=cv2.INTER_NEAREST)
            for c in range(len(FINGER_LABELS))
        ], axis=0)
        return img_t, torch.from_numpy(masks_r)

    # ------------------------------------------------------------------
    def _make_masks(self, polygon_groups, orig_w, orig_h):
        ms    = self.image_size
        masks = np.zeros((len(FINGER_LABELS), ms, ms), dtype=np.float32)
        for pg in polygon_groups:
            label = pg['label']
            if label not in FINGER_LABELS:
                continue
            ch = FINGER_LABELS.index(label)
            for poly in pg['polygons']:
                if poly.get('hole', False):
                    continue
                pts = np.array([[p['x'] / orig_w * ms,
                                 p['y'] / orig_h * ms]
                                for p in poly['points']], dtype=np.int32)
                cv2.fillPoly(masks[ch], [pts], 1.0)
        return masks

    def _flip(self, arr, masks):
        """Horizontal flip with L2↔R2 and L4↔R4 swap."""
        arr_f   = arr[:, ::-1].copy()
        m       = masks
        masks_f = np.zeros_like(m)
        masks_f[0] = m[2, :, ::-1].copy()   # L2 ← R2
        masks_f[2] = m[0, :, ::-1].copy()   # R2 ← L2
        masks_f[1] = m[3, :, ::-1].copy()   # L4 ← R4
        masks_f[3] = m[1, :, ::-1].copy()   # R4 ← L4
        return arr_f, masks_f

    def _rotate(self, arr, masks, max_angle=12):
        angle = np.random.uniform(-max_angle, max_angle)
        h, w  = arr.shape
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        arr_r = cv2.warpAffine(arr, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        ms    = masks.shape[1]
        Mm    = cv2.getRotationMatrix2D((ms / 2, ms / 2), angle, 1.0)
        masks_r = np.stack([
            cv2.warpAffine(masks[c], Mm, (ms, ms),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
            for c in range(masks.shape[0])
        ], axis=0)
        return arr_r, masks_r

    def _elastic_deform(self, arr, masks, alpha=30, sigma=5):
        h, w  = arr.shape
        dx    = cv2.GaussianBlur((np.random.rand(h, w)*2-1).astype(np.float32),
                                 (int(sigma*6)|1, int(sigma*6)|1), sigma) * alpha
        dy    = cv2.GaussianBlur((np.random.rand(h, w)*2-1).astype(np.float32),
                                 (int(sigma*6)|1, int(sigma*6)|1), sigma) * alpha
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        map_x  = (xs + dx).astype(np.float32)
        map_y  = (ys + dy).astype(np.float32)
        arr_d  = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        ms = masks.shape[1]
        dx_m = cv2.resize(dx, (ms, ms))
        dy_m = cv2.resize(dy, (ms, ms))
        xs_m, ys_m = np.meshgrid(np.arange(ms), np.arange(ms))
        map_xm = (xs_m + dx_m).astype(np.float32)
        map_ym = (ys_m + dy_m).astype(np.float32)
        masks_d = np.stack([
            cv2.remap(masks[c], map_xm, map_ym, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)
            for c in range(masks.shape[0])
        ], axis=0)
        return arr_d, masks_d

    def _perspective(self, arr, masks, max_shift=0.05):
        h, w  = arr.shape
        shift = max_shift * min(h, w)
        src   = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst   = src + np.random.uniform(-shift, shift, src.shape).astype(np.float32)
        M     = cv2.getPerspectiveTransform(src, dst)
        arr_p = cv2.warpPerspective(arr, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
        ms = masks.shape[1]
        src_m = np.float32([[0,0],[ms,0],[ms,ms],[0,ms]])
        dst_m = (dst / np.array([w, h]) * ms).astype(np.float32)
        Mm    = cv2.getPerspectiveTransform(src_m, dst_m)
        masks_p = np.stack([
            cv2.warpPerspective(masks[c], Mm, (ms, ms),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
            for c in range(masks.shape[0])
        ], axis=0)
        return arr_p, masks_p

    def _brightness(self, arr):
        alpha = np.random.uniform(0.8, 1.2)
        beta  = np.random.randint(-25, 25)
        return np.clip(alpha * arr.astype(np.float32) + beta, 0, 255).astype(np.uint8)

# =============================================================================
# TRAINING
# =============================================================================

def train():
    start  = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    full_ds  = FingerSegDataset(CONFIG['annotations_file'], CONFIG['images_dir'],
                                CONFIG['image_size'], augment=False)
    n_train  = int(CONFIG['train_split'] * len(full_ds))
    n_val    = len(full_ds) - n_train
    base_idx, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    aug_ds = FingerSegDataset(CONFIG['annotations_file'], CONFIG['images_dir'],
                              CONFIG['image_size'], augment=True)
    aug_ds.samples = [full_ds.samples[i] for i in base_idx.indices]

    train_loader = DataLoader(aug_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,  batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=0)
    print(f"Train: {len(aug_ds)}  Val: {n_val}")

    model = FingerSegUNet(num_classes=CONFIG['num_fingers'], pretrained=True).to(device)
    # Pretrained encoder gets 10× lower LR — it already knows useful features
    encoder_params = (list(model.enc0.parameters()) + list(model.enc1.parameters()) +
                      list(model.enc2.parameters()) + list(model.enc3.parameters()) +
                      list(model.enc4.parameters()))
    decoder_params = (list(model.up4.parameters()) + list(model.up3.parameters()) +
                      list(model.up2.parameters()) + list(model.up1.parameters()) +
                      list(model.up0.parameters()) + list(model.head.parameters()))
    optimizer = torch.optim.Adam([
        {'params': encoder_params, 'lr': CONFIG['lr'] * 0.1},
        {'params': decoder_params, 'lr': CONFIG['lr']},
    ])
    scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    pos_weight = torch.tensor([CONFIG['pos_weight']], device=device)

    def combined_loss(logits, masks):
        ce   = F.binary_cross_entropy_with_logits(logits, masks, pos_weight=pos_weight)
        prob = torch.sigmoid(logits)
        inter = (prob * masks).sum(dim=(2, 3))
        dice = 1 - (2 * inter + 1) / (prob.sum(dim=(2,3)) + masks.sum(dim=(2,3)) + 1)
        return 0.6 * ce + 0.4 * dice.mean()

    os.makedirs('trained_model', exist_ok=True)
    best_val    = float('inf')
    start_epoch = 0
    resume_path = 'trained_model/resume_seg.pth'

    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optimizer_state'])
        scheduler.load_state_dict(ck['scheduler_state'])
        best_val    = ck['best_val']
        start_epoch = ck['epoch']
        print(f"  Resumed from epoch {start_epoch}/{CONFIG['epochs']}  best={best_val:.5f}")

    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        tr_loss = 0.0
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} — training...", flush=True)
        for batch_i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            if (batch_i + 1) % 5 == 0:
                print(f"  batch {batch_i+1}/{len(train_loader)}  loss={loss.item():.4f}",
                      flush=True)
        scheduler.step()

        model.eval()
        vl_loss = 0.0
        iou_sum = 0.0; iou_count = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits  = model(imgs)
                vl_loss += combined_loss(logits, masks).item()
                pred = (torch.sigmoid(logits) > 0.5).float()
                tp = (pred * masks).sum(dim=(2, 3))
                fp = (pred * (1 - masks)).sum(dim=(2, 3))
                fn = ((1 - pred) * masks).sum(dim=(2, 3))
                iou = (tp / (tp + fp + fn + 1e-6)).mean().item()
                iou_sum += iou; iou_count += 1

        nb_t = len(train_loader); nb_v = max(len(val_loader), 1)
        mean_iou = iou_sum / max(iou_count, 1)
        saved = ''
        if vl_loss / nb_v < best_val:
            best_val = vl_loss / nb_v
            torch.save({'model_state_dict': model.state_dict(),
                        'val_loss':          best_val,
                        'finger_labels':     FINGER_LABELS,
                        'config':            CONFIG},
                       'trained_model/best_seg_model.pth')
            saved = ' ✓ saved'

        # Save resume checkpoint every epoch so training can be paused and continued
        torch.save({'model_state':     model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_val':        best_val,
                    'epoch':           epoch + 1}, resume_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{CONFIG['epochs']}: "
                  f"train={tr_loss/nb_t:.4f}  val={vl_loss/nb_v:.4f}  "
                  f"IoU={mean_iou*100:.2f}%{saved}")

    # Training complete — remove resume checkpoint so a fresh run starts clean
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"\nDone in {(time.time()-start)/60:.1f} min.  Best val: {best_val:.5f}")
    print("Saved: trained_model/best_seg_model.pth")

    # Export to ONNX for visualization in Netron (netron.app)
    dummy = torch.zeros(1, 1, CONFIG['image_size'], CONFIG['image_size']).to(device)
    torch.onnx.export(model, dummy, 'trained_model/seg_model.onnx',
                      input_names=['image'], output_names=['masks'],
                      opset_version=11)
    print("Exported: trained_model/seg_model.onnx  (open at netron.app)")


if __name__ == '__main__':
    try:
        train()
    except Exception:
        import traceback; traceback.print_exc()
