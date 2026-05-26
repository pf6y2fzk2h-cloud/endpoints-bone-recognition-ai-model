#!/usr/bin/env python3.12
"""
Stage 2 — Per-Finger Keypoint Detector on Crops
=================================================
Trains on crops defined by the SEGMENTATION MODEL (not GT polygons),
so training and inference see the exact same type of crops.
Falls back to GT polygon crops if the seg model misses a finger.

Output: trained_model/best_kp_model.pth
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
    'seg_model_path':   'trained_model/best_seg_model.pth',
    'kp_model_path':    'trained_model/best_kp_model.pth',
    'seg_image_size':   256,    # must match train_finger_seg.py image_size
    'seg_threshold':    0.2,
    'crop_size':        256,
    'heatmap_size':     64,
    'sigma':            1.0,
    'sigma_hard':       1.5,
    'pad_fraction':     0.10,   # matches predict_cascade.py
    'train_split':      0.85,
    'batch_size':       8,
    'lr':               0.001,
    'epochs':           50,
}

FINGER_LABELS = ['L2', 'L4', 'R2', 'R4']

FINGER_KP_INDICES = {
    'L2': list(range(0,  6)),
    'L4': list(range(6,  12)),
    'R2': list(range(12, 18)),
    'R4': list(range(18, 24)),
}

ALL_KP_LABELS = [
    'L21-','L21_','L22-','L22_','L23-','L23_',
    'L41-','L41_','L42-','L42_','L43-','L43_',
    'R21-','R21_','R22-','R22_','R23-','R23_',
    'R41-','R41_','R42-','R42_','R43-','R43_',
]

HARD_KP_INDICES_IN_CROP = {5}   # bone-3 bottom = hardest remaining endpoint

# =============================================================================
# SEG MODEL  (must match train_finger_seg.py exactly)
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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))


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
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        try:
            bb = tv_models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            bb = tv_models.resnet18(pretrained=pretrained)
        self.enc0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            bb.bn1, nn.ReLU(inplace=True),
        )
        if pretrained:
            with torch.no_grad():
                self.enc0[0].weight.data = bb.conv1.weight.data.mean(dim=1, keepdim=True)
        self.pool = bb.maxpool
        self.enc1 = bb.layer1
        self.enc2 = bb.layer2
        self.enc3 = bb.layer3
        self.enc4 = bb.layer4
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
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.enc4(e3)
        d  = self.up4(b,  e3)
        d  = self.up3(d,  e2)
        d  = self.up2(d,  e1)
        d  = self.up1(d,  e0)
        d  = self.up0(d)
        return self.head(d)

# =============================================================================
# KP MODEL
# =============================================================================

class FingerKpModel(nn.Module):
    def __init__(self, num_kp=6):
        super().__init__()
        self.enc1   = ConvBlock(1,   32)
        self.pool1  = nn.MaxPool2d(2)
        self.enc2   = ConvBlock(32,  64)
        self.pool2  = nn.MaxPool2d(2)
        self.enc3   = ConvBlock(64,  128)
        self.pool3  = nn.MaxPool2d(2)
        self.bottle = ConvBlock(128, 256)
        self.up3    = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3   = ConvBlock(256, 128)
        self.up2    = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2   = ConvBlock(128, 64)
        self.up1    = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1   = ConvBlock(64, 32)
        self.head   = nn.Conv2d(32, num_kp, 1)

    @staticmethod
    def _up(layer, x, ref):
        x = layer(x)
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottle(self.pool3(e3))
        d3 = self.dec3(torch.cat([self._up(self.up3, b,  e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(self.up2, d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(self.up1, d2, e1), e1], dim=1))
        return self.head(d1)

# =============================================================================
# CROP UTILITIES
# =============================================================================

def apply_clahe(arr):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr)


def polygon_bbox(polygon_groups, finger_label, orig_w, orig_h):
    for pg in polygon_groups:
        if pg['label'] != finger_label:
            continue
        xs, ys = [], []
        for poly in pg['polygons']:
            if poly.get('hole', False):
                continue
            for p in poly['points']:
                xs.append(p['x']); ys.append(p['y'])
        if xs:
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def make_square_crop(arr, bbox_px, pad_frac):
    """bbox_px in original image pixel coords. Returns (crop, (cx1,cy1,cx2,cy2))."""
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = bbox_px
    pw = (x2 - x1) * pad_frac; ph = (y2 - y1) * pad_frac
    x1 -= pw; y1 -= ph; x2 += pw; y2 += ph
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    x1 = cx - side / 2; x2 = cx + side / 2
    y1 = cy - side / 2; y2 = cy + side / 2
    cx1 = int(max(0, x1)); cy1 = int(max(0, y1))
    cx2 = int(min(w, x2)); cy2 = int(min(h, y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None
    return arr[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


def mask_bbox_to_image_coords(bbox_model, model_size, orig_w, orig_h):
    """Scale a bbox from model space (e.g. 256×256) back to original image pixels."""
    sx = orig_w / model_size; sy = orig_h / model_size
    x1, y1, x2, y2 = bbox_model
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def fix_finger_assignment(seg_probs, threshold):
    """Swap L2/L4 or R2/R4 if their centroids are anatomically reversed."""
    def cx(ch):
        pts = np.where(seg_probs[ch] > threshold)
        return float(pts[1].mean()) if len(pts[1]) > 0 else None
    l2_x, l4_x = cx(0), cx(1)
    if l2_x is not None and l4_x is not None and l2_x < l4_x:
        seg_probs[[0, 1]] = seg_probs[[1, 0]]
    r2_x, r4_x = cx(2), cx(3)
    if r2_x is not None and r4_x is not None and r2_x > r4_x:
        seg_probs[[2, 3]] = seg_probs[[3, 2]]
    return seg_probs

# =============================================================================
# PRE-COMPUTE SEG CROPS
# =============================================================================

def compute_seg_crops(seg_model, image_names, images_dir, config, device):
    """
    Run the seg model on every training image and return a dict:
        {(image_name, finger_label): (cx1, cy1, cx2, cy2) or None}

    This is done once before training so the dataset can use seg-predicted
    crops instead of GT polygon crops — matching exactly what predict_cascade.py
    does at inference time.
    """
    seg_size  = config['seg_image_size']
    pad_frac  = config['pad_fraction']
    threshold = config['seg_threshold']
    crop_boxes = {}

    seg_model.eval()
    print(f"  Pre-computing seg crops for {len(image_names)} images...", flush=True)

    with torch.no_grad():
        for name in image_names:
            path = os.path.join(images_dir, name)
            arr  = np.array(Image.open(path).convert('L'), dtype=np.uint8)
            orig_h, orig_w = arr.shape

            enhanced = apply_clahe(arr)
            resized  = cv2.resize(enhanced, (seg_size, seg_size))
            t = torch.from_numpy(resized.astype(np.float32) / 255.0)
            t = t.unsqueeze(0).unsqueeze(0).to(device)   # (1,1,H,W)

            logits = seg_model(t)
            probs  = torch.sigmoid(logits)[0].cpu().numpy()  # (4,H,W)
            probs  = fix_finger_assignment(probs, threshold)

            for ch, fl in enumerate(FINGER_LABELS):
                key    = (name, fl)
                binary = (probs[ch] > threshold).astype(np.uint8)
                if binary.sum() == 0:
                    crop_boxes[key] = None
                    continue

                # Largest connected component
                n_lb, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
                if n_lb <= 1:
                    crop_boxes[key] = None
                    continue
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                bx1 = int(stats[largest, cv2.CC_STAT_LEFT])
                by1 = int(stats[largest, cv2.CC_STAT_TOP])
                bx2 = bx1 + int(stats[largest, cv2.CC_STAT_WIDTH])
                by2 = by1 + int(stats[largest, cv2.CC_STAT_HEIGHT])

                bbox_img = mask_bbox_to_image_coords(
                    (bx1, by1, bx2, by2), seg_size, orig_w, orig_h)
                _, box = make_square_crop(arr, bbox_img, pad_frac)
                if box is None:
                    crop_boxes[key] = None
                    continue

                # Also store the finger mask cropped to this box (in seg model space)
                # so the kp dataset can apply it to isolate the finger
                cx1, cy1, cx2, cy2 = box
                mx1 = max(0, int(cx1 / orig_w * seg_size))
                my1 = max(0, int(cy1 / orig_h * seg_size))
                mx2 = min(seg_size, int(cx2 / orig_w * seg_size))
                my2 = min(seg_size, int(cy2 / orig_h * seg_size))
                mask_crop = (probs[ch, my1:my2, mx1:mx2].copy()
                             if mx2 > mx1 and my2 > my1 else probs[ch].copy())
                crop_boxes[key] = {'box': box, 'mask': mask_crop}

    found = sum(1 for v in crop_boxes.values() if v is not None)
    print(f"  Seg crops: {found}/{len(crop_boxes)} found  "
          f"({len(crop_boxes)-found} fell back to GT polygons)", flush=True)
    return crop_boxes

# =============================================================================
# DATASET
# =============================================================================

class FingerCropDataset(Dataset):
    def __init__(self, annotations_file, images_dir, config,
                 augment=False, seg_crop_boxes=None):
        with open(annotations_file) as f:
            data = json.load(f)
        self.images_dir    = images_dir
        self.config        = config
        self.augment       = augment
        self.seg_crop_boxes = seg_crop_boxes or {}

        self.samples = []
        for img_data in data:
            poly_labels = {pg['label'] for pg in img_data.get('polygonGroups', [])}
            has_kp      = len(img_data.get('points', [])) > 0
            path        = os.path.join(images_dir, img_data['name'])
            if not (has_kp and os.path.exists(path)):
                continue
            for fl in FINGER_LABELS:
                if fl in poly_labels:
                    self.samples.append((img_data, fl))

        print(f"  FingerCropDataset: {len(self.samples)} crops  augment={augment}  "
              f"seg_crops={'yes' if seg_crop_boxes else 'no (GT polygon fallback)'}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_data, finger_label = self.samples[idx]
        arr = np.array(Image.open(
            os.path.join(self.images_dir, img_data['name'])).convert('L'),
            dtype=np.uint8)
        orig_h, orig_w = arr.shape
        arr = apply_clahe(arr)

        # ---- Use seg crop if available, else fall back to GT polygon ----
        key       = (img_data['name'], finger_label)
        seg_entry = self.seg_crop_boxes.get(key)  # None, or {'box':..., 'mask':...}
        seg_mask  = None
        crop_box  = None

        if seg_entry is not None:
            box = seg_entry['box'] if isinstance(seg_entry, dict) else seg_entry
            seg_mask = seg_entry.get('mask') if isinstance(seg_entry, dict) else None
            cx1, cy1, cx2, cy2 = box
            crop = arr[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crop_box = box
            else:
                seg_mask = None   # invalid crop — fall back

        if crop_box is None:
            bbox = polygon_bbox(img_data['polygonGroups'], finger_label, orig_w, orig_h)
            if bbox is None:
                cs = self.config['crop_size']
                return torch.zeros(1, cs, cs), torch.zeros(6, cs, cs), torch.zeros(6)
            crop, (cx1, cy1, cx2, cy2) = make_square_crop(
                arr, bbox, self.config['pad_fraction'])
            if crop is None:
                cs = self.config['crop_size']
                return torch.zeros(1, cs, cs), torch.zeros(6, cs, cs), torch.zeros(6)

        # ---- Keypoints relative to crop ----
        crop_h, crop_w = crop.shape
        kps = np.zeros((6, 2), dtype=np.float32)
        vis = np.zeros(6, dtype=np.float32)
        points_dict = {p['label']: (p['x'], p['y']) for p in img_data.get('points', [])}
        for j, global_idx in enumerate(FINGER_KP_INDICES[finger_label]):
            label = ALL_KP_LABELS[global_idx]
            if label not in points_dict:
                continue
            gx, gy = points_dict[label]
            rx = (gx - cx1) / max(crop_w, 1)
            ry = (gy - cy1) / max(crop_h, 1)
            if 0.0 <= rx <= 1.0 and 0.0 <= ry <= 1.0:
                kps[j] = [rx, ry]; vis[j] = 1.0

        # ---- Augmentation ----
        if self.augment:
            if np.random.rand() < 0.5:
                crop, kps, vis = self._rotate(crop, kps, vis)
            if np.random.rand() < 0.4:
                crop, kps, vis = self._elastic_deform(crop, kps, vis)
            if np.random.rand() < 0.3:
                crop, kps, vis = self._perspective(crop, kps, vis)
            if np.random.rand() < 0.5:
                crop = self._brightness(crop)

        cs     = self.config['crop_size']
        crop_r = cv2.resize(crop, (cs, cs), interpolation=cv2.INTER_LINEAR)

        # Apply seg mask — isolates the target finger, zeroes out neighbours
        if seg_mask is not None:
            m     = cv2.resize(seg_mask, (cs, cs), interpolation=cv2.INTER_LINEAR)
            kern  = np.ones((9, 9), np.uint8)
            m_dil = cv2.dilate((m > 0.15).astype(np.uint8),
                               kern, iterations=3).astype(np.float32)
            crop_r = np.clip(crop_r.astype(np.float32) * m_dil, 0, 255).astype(np.uint8)

        crop_t = torch.from_numpy(crop_r.astype(np.float32) / 255.0).unsqueeze(0)
        return crop_t, self._make_heatmaps(kps, vis), torch.from_numpy(vis)

    def _make_heatmaps(self, kps, vis):
        hs   = self.config['heatmap_size']
        maps = np.zeros((6, hs, hs), dtype=np.float32)
        for j in range(6):
            if vis[j] < 0.5: continue
            x = int(kps[j, 0] * hs); y = int(kps[j, 1] * hs)
            if not (0 <= x < hs and 0 <= y < hs): continue
            sigma = CONFIG['sigma_hard'] if j in HARD_KP_INDICES_IN_CROP else CONFIG['sigma']
            sz = int(sigma * 6)
            for yy in range(max(0, y-sz), min(hs, y+sz+1)):
                for xx in range(max(0, x-sz), min(hs, x+sz+1)):
                    g = np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2))
                    maps[j, yy, xx] = max(maps[j, yy, xx], g)
        return torch.from_numpy(maps)

    def _rotate(self, crop, kps, vis, max_angle=15):
        angle  = np.random.uniform(-max_angle, max_angle)
        h, w   = crop.shape
        M      = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        crop_r = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        rad = np.radians(-angle); cos_a = np.cos(rad); sin_a = np.sin(rad)
        kps_r = kps.copy(); vis_r = vis.copy()
        for j in range(6):
            if vis[j] < 0.5: continue
            x = kps[j,0]-0.5; y = kps[j,1]-0.5
            nx = cos_a*x - sin_a*y + 0.5; ny = sin_a*x + cos_a*y + 0.5
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                kps_r[j] = [nx, ny]
            else:
                vis_r[j] = 0.0
        return crop_r, kps_r, vis_r

    def _elastic_deform(self, crop, kps, vis, alpha=20, sigma=4):
        h, w  = crop.shape
        dx    = cv2.GaussianBlur((np.random.rand(h, w)*2-1).astype(np.float32),
                                 (int(sigma*6)|1, int(sigma*6)|1), sigma) * alpha
        dy    = cv2.GaussianBlur((np.random.rand(h, w)*2-1).astype(np.float32),
                                 (int(sigma*6)|1, int(sigma*6)|1), sigma) * alpha
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        map_x  = (xs + dx).astype(np.float32)
        map_y  = (ys + dy).astype(np.float32)
        crop_d = cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        kps_d = kps.copy(); vis_d = vis.copy()
        for j in range(6):
            if vis[j] < 0.5: continue
            ix = int(np.clip(kps[j,0]*w, 0, w-1))
            iy = int(np.clip(kps[j,1]*h, 0, h-1))
            nx = kps[j,0]*w - dx[iy, ix]
            ny = kps[j,1]*h - dy[iy, ix]
            if 0 <= nx <= w and 0 <= ny <= h:
                kps_d[j] = [nx/w, ny/h]
            else:
                vis_d[j] = 0.0
        return crop_d, kps_d, vis_d

    def _perspective(self, crop, kps, vis, max_shift=0.05):
        h, w  = crop.shape
        shift = max_shift * min(h, w)
        src   = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst   = src + np.random.uniform(-shift, shift, src.shape).astype(np.float32)
        M     = cv2.getPerspectiveTransform(src, dst)
        crop_p = cv2.warpPerspective(crop, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        kps_p = kps.copy(); vis_p = vis.copy()
        for j in range(6):
            if vis[j] < 0.5: continue
            pt  = np.array([[[kps[j,0]*w, kps[j,1]*h]]], dtype=np.float32)
            pt2 = cv2.perspectiveTransform(pt, M)[0,0]
            if 0 <= pt2[0] <= w and 0 <= pt2[1] <= h:
                kps_p[j] = [pt2[0]/w, pt2[1]/h]
            else:
                vis_p[j] = 0.0
        return crop_p, kps_p, vis_p

    def _brightness(self, arr):
        alpha = np.random.uniform(0.8, 1.2)
        beta  = np.random.randint(-25, 25)
        return np.clip(alpha*arr.astype(np.float32)+beta, 0, 255).astype(np.uint8)

# =============================================================================
# LOSS
# =============================================================================

_KP_WEIGHTS = torch.ones(6)
for _j in HARD_KP_INDICES_IN_CROP: _KP_WEIGHTS[_j] = 4.0   # bone-3 bottom
for _j in [1, 3]:                   _KP_WEIGHTS[_j] = 3.0   # bone-1,2 bottoms


def keypoint_loss(pred, target, vis):
    mse = F.mse_loss(pred, target, reduction='none')
    v   = vis.unsqueeze(-1).unsqueeze(-1)
    w   = _KP_WEIGHTS.to(pred.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return (mse * v * w).sum() / ((v * w).sum() + 1e-6)

# =============================================================================
# TRAINING
# =============================================================================

def train():
    start  = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load seg model to pre-compute crops ----
    seg_crop_boxes = {}
    if os.path.exists(CONFIG['seg_model_path']):
        print("Loading seg model for crop pre-computation...", flush=True)
        seg_ck    = torch.load(CONFIG['seg_model_path'], map_location=device)
        seg_model = FingerSegUNet(4, pretrained=False).to(device)
        seg_model.load_state_dict(seg_ck['model_state_dict'])

        # Collect all unique image names from the dataset
        with open(CONFIG['annotations_file']) as f:
            all_data = json.load(f)
        all_names = [d['name'] for d in all_data
                     if os.path.exists(os.path.join(CONFIG['images_dir'], d['name']))]
        seg_crop_boxes = compute_seg_crops(
            seg_model, all_names, CONFIG['images_dir'], CONFIG, device)
        del seg_model  # free memory
    else:
        print("WARNING: seg model not found — using GT polygon crops (training/inference mismatch)")

    # ---- Build datasets ----
    full_ds   = FingerCropDataset(CONFIG['annotations_file'], CONFIG['images_dir'],
                                  CONFIG, augment=False, seg_crop_boxes=seg_crop_boxes)
    all_imgs  = list({s[0]['name'] for s in full_ds.samples})
    n_train   = int(CONFIG['train_split'] * len(all_imgs))
    rng       = np.random.default_rng(42)
    rng.shuffle(all_imgs)
    train_imgs = set(all_imgs[:n_train])
    val_imgs   = set(all_imgs[n_train:])

    aug_ds = FingerCropDataset(CONFIG['annotations_file'], CONFIG['images_dir'],
                               CONFIG, augment=True, seg_crop_boxes=seg_crop_boxes)
    aug_ds.samples = [(d, fl) for d, fl in aug_ds.samples if d['name'] in train_imgs]

    val_ds = FingerCropDataset(CONFIG['annotations_file'], CONFIG['images_dir'],
                               CONFIG, augment=False, seg_crop_boxes=seg_crop_boxes)
    val_ds.samples = [(d, fl) for d, fl in val_ds.samples if d['name'] in val_imgs]

    print(f"  Train crops: {len(aug_ds)}  Val crops: {len(val_ds)}", flush=True)

    train_loader = DataLoader(aug_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,  batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=0)

    model     = FingerKpModel(num_kp=6).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.99)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    os.makedirs('trained_model', exist_ok=True)
    best_val    = float('inf')
    start_epoch = 0

    # ---- Resume from checkpoint if available ----
    resume_path = 'trained_model/resume_kp.pth'
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        best_val    = ck['best_val']
        start_epoch = ck['epoch']  # epoch we stopped AT (next run starts from +1)
        print(f"  Resumed from epoch {start_epoch}/{CONFIG['epochs']}  "
              f"best_val={best_val:.5f}", flush=True)

    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        tr_loss = 0.0
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} — training...", flush=True)
        for batch_i, (crops, heatmaps, vis) in enumerate(train_loader):
            crops, heatmaps, vis = crops.to(device), heatmaps.to(device), vis.to(device)
            optimizer.zero_grad()
            pred = model(crops)
            if pred.shape[-1] != heatmaps.shape[-1]:
                heatmaps = F.interpolate(heatmaps, size=pred.shape[-2:],
                                         mode='bilinear', align_corners=False)
            loss = keypoint_loss(pred, heatmaps, vis)
            loss.backward(); optimizer.step()
            tr_loss += loss.item()
            if (batch_i + 1) % 5 == 0:
                print(f"  batch {batch_i+1}/{len(train_loader)}  loss={loss.item():.5f}",
                      flush=True)
        scheduler.step()

        model.eval()
        vl_loss = 0.0
        dist_sum = 0.0; dist_count = 0
        with torch.no_grad():
            for crops, heatmaps, vis in val_loader:
                crops, heatmaps, vis = crops.to(device), heatmaps.to(device), vis.to(device)
                pred = model(crops)
                if pred.shape[-1] != heatmaps.shape[-1]:
                    heatmaps = F.interpolate(heatmaps, size=pred.shape[-2:],
                                             mode='bilinear', align_corners=False)
                vl_loss += keypoint_loss(pred, heatmaps, vis).item()

                # Mean distance error in pixels (crop space)
                pred_sig = torch.sigmoid(pred).cpu().numpy()
                hm_np    = heatmaps.cpu().numpy()
                vis_np   = vis.cpu().numpy()
                hs = pred_sig.shape[-1]
                for b in range(pred_sig.shape[0]):
                    for j in range(6):
                        if vis_np[b, j] < 0.5:
                            continue
                        # Predicted location
                        py, px = np.unravel_index(pred_sig[b, j].argmax(), (hs, hs))
                        # True location (argmax of target heatmap)
                        ty, tx = np.unravel_index(hm_np[b, j].argmax(), (hs, hs))
                        dist_sum   += np.hypot(px - tx, py - ty)
                        dist_count += 1

        nb_t = len(train_loader); nb_v = max(len(val_loader), 1)
        mean_dist = dist_sum / max(dist_count, 1)
        saved = ''
        if vl_loss / nb_v < best_val:
            best_val = vl_loss / nb_v
            torch.save({'model_state_dict': model.state_dict(),
                        'val_loss':          best_val,
                        'config':            CONFIG},
                       'trained_model/best_kp_model.pth')
            saved = ' ✓ saved'

        # Save resume checkpoint every epoch so training can be paused and continued
        torch.save({'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch':                epoch + 1,
                    'best_val':             best_val},
                   resume_path)

        print(f"  Epoch {epoch+1:3d}/{CONFIG['epochs']}: "
              f"train={tr_loss/nb_t:.5f}  val={vl_loss/nb_v:.5f}  "
              f"MDE={mean_dist:.2f}px{saved}", flush=True)

    # Training complete — remove resume checkpoint so a fresh run starts clean
    if os.path.exists(resume_path):
        os.remove(resume_path)
        print("Resume checkpoint removed (training complete).")

    print(f"\nDone in {(time.time()-start)/60:.1f} min.  Best val: {best_val:.6f}")
    print("Saved: trained_model/best_kp_model.pth")


if __name__ == '__main__':
    try:
        train()
    except Exception:
        import traceback; traceback.print_exc()
