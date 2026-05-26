#!/usr/bin/env python3.12
"""
Cascade Prediction: Finger Segmentation → Crop → Keypoints
===========================================================

Stage 1: FingerSegUNet predicts L2/L4/R2/R4 masks (512×512)
Stage 2: For each finger mask, extract a square crop from the original image
Stage 3: FingerKpModel predicts 6 keypoints within that crop
Stage 4: Transform crop-space keypoints back to original image coordinates

This two-stage approach is far more robust than predicting all 24 keypoints
in the full image because each sub-model solves a much simpler problem.
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
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    'seg_model_path':  'trained_model/best_seg_model.pth',
    'kp_model_path':   'trained_model/best_kp_model.pth',
    'input_folder':    '../new_xrays',
    'output_folder':   'predictions',
    'image_size':      256,
    'crop_size':       256,
    'seg_threshold':   0.2,
    'kp_threshold':    0.0,
    'kp_threshold_hard': 0.0,
    'pad_fraction':    0.10,
    'max_images':      30,
}

FINGER_LABELS   = ['L2', 'L4', 'R2', 'R4']
FINGER_KP_NAMES = {
    'L2': ['L21-','L21_','L22-','L22_','L23-','L23_'],
    'L4': ['L41-','L41_','L42-','L42_','L43-','L43_'],
    'R2': ['R21-','R21_','R22-','R22_','R23-','R23_'],
    'R4': ['R41-','R41_','R42-','R42_','R43-','R43_'],
}
ALL_KP_LABELS = (FINGER_KP_NAMES['L2'] + FINGER_KP_NAMES['L4'] +
                 FINGER_KP_NAMES['R2'] + FINGER_KP_NAMES['R4'])

FINGER_COLORS = {
    'L2': (255,   0, 255),
    'L4': (255, 165,   0),
    'R2': (  0, 255, 255),
    'R4': (  0, 255,   0),
}

# =============================================================================
# MODELS  (must match train_finger_seg.py / train_finger_kp.py exactly)
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
# PREPROCESSING
# =============================================================================

def apply_clahe(arr):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr)


def preprocess_full(pil_gray, size=512):
    """Prepare full image for the seg model: CLAHE → resize → [0,1] tensor."""
    arr = apply_clahe(np.array(pil_gray, dtype=np.uint8))
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def preprocess_crop(crop_arr, size=128, mask=None):
    """
    Prepare a finger crop for the kp model: CLAHE → resize → mask → [0,1] tensor.
    mask: optional (H, W) float32 probability map in the same space as crop_arr.
          Applied after resize so the kp model only sees the target finger.
    """
    arr = apply_clahe(crop_arr)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        m     = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
        kern  = np.ones((9, 9), np.uint8)
        m_dil = cv2.dilate((m > 0.15).astype(np.uint8),
                           kern, iterations=3).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) * m_dil, 0, 255).astype(np.uint8)
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

# =============================================================================
# CROP EXTRACTION FROM MASK
# =============================================================================

def mask_to_bbox(mask_np, threshold=0.3):
    """
    Given a (H, W) probability mask, find the bounding box of the largest
    connected component above threshold.
    Returns (x1, y1, x2, y2) in mask pixel coordinates, or None.
    """
    binary = (mask_np > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return None

    # Largest component (skip background = label 0)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x1 = int(stats[largest, cv2.CC_STAT_LEFT])
    y1 = int(stats[largest, cv2.CC_STAT_TOP])
    x2 = x1 + int(stats[largest, cv2.CC_STAT_WIDTH])
    y2 = y1 + int(stats[largest, cv2.CC_STAT_HEIGHT])
    return (x1, y1, x2, y2)


def fix_finger_assignment(seg_probs, threshold=0.2):
    """
    Anatomical sanity check on segmentation masks.

    Known hand anatomy (dorsal view, fingers pointing up):
      Left hand : L2 (index) centroid is to the RIGHT of L4 (ring)
      Right hand: R2 (index) centroid is to the LEFT  of R4 (ring)

    If the segmentation swapped a pair, swap the masks back.
    FINGER_LABELS order: [L2=0, L4=1, R2=2, R4=3]
    """
    def cx(ch):
        m = seg_probs[ch]
        pts = np.where(m > threshold)
        return float(pts[1].mean()) if len(pts[1]) > 0 else None

    l2_x, l4_x = cx(0), cx(1)
    if l2_x is not None and l4_x is not None and l2_x < l4_x:
        print("  [anatomy] L2/L4 swapped — correcting", flush=True)
        seg_probs[[0, 1]] = seg_probs[[1, 0]]

    r2_x, r4_x = cx(2), cx(3)
    if r2_x is not None and r4_x is not None and r2_x > r4_x:
        print("  [anatomy] R2/R4 swapped — correcting", flush=True)
        seg_probs[[2, 3]] = seg_probs[[3, 2]]

    return seg_probs


def make_square_crop(arr, bbox_model, model_size, orig_size, pad_frac=0.25):
    """
    bbox_model : (x1,y1,x2,y2) in seg model (512×512) coordinate space
    orig_size  : (orig_w, orig_h) of the original image
    Returns:
        crop_arr  : H×W numpy array from original image
        crop_box  : (cx1,cy1,cx2,cy2) in ORIGINAL image pixels
    """
    orig_w, orig_h = orig_size
    sx = orig_w / model_size; sy = orig_h / model_size

    # Scale bbox back to original image space
    x1 = bbox_model[0] * sx; y1 = bbox_model[1] * sy
    x2 = bbox_model[2] * sx; y2 = bbox_model[3] * sy

    # Add padding
    pw = (x2 - x1) * pad_frac; ph = (y2 - y1) * pad_frac
    x1 -= pw; y1 -= ph; x2 += pw; y2 += ph

    # Make square around center
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    x1 = cx - side / 2; x2 = cx + side / 2
    y1 = cy - side / 2; y2 = cy + side / 2

    # Clip to image
    cx1 = int(max(0, x1)); cy1 = int(max(0, y1))
    cx2 = int(min(orig_w, x2)); cy2 = int(min(orig_h, y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None

    return arr[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)

# =============================================================================
# KEYPOINT EXTRACTION FROM CROP HEATMAPS
# =============================================================================

def predict_crop_tta(model, crop_t, device):
    """
    Test-time augmentation for a single finger crop.
    Runs the kp model on the original crop AND a horizontally flipped copy,
    then averages the heatmaps. No channel swap needed — it's one finger,
    flipping just mirrors it left-right.
    """
    with torch.no_grad():
        hm_orig = model(crop_t.to(device))[0]          # (6, H, W)
        hm_flip = model(crop_t.flip(-1).to(device))[0] # predict on flipped
    hm_flip = hm_flip.flip(-1)                          # flip heatmaps back
    return (hm_orig + hm_flip) / 2


def extract_crop_keypoints(heatmap_tensor, kp_names, threshold, hard_threshold):
    """
    heatmap_tensor: (6, H, W) raw logits from the kp model
    Returns: kps (6,2) in [0,1] crop-relative coords, conf (6,)
    """
    hms = torch.sigmoid(heatmap_tensor).cpu().numpy()  # (6, H, W)
    kps  = np.zeros((6, 2), dtype=np.float32)
    conf = np.zeros(6, dtype=np.float32)

    for j in range(6):
        hm  = hms[j]
        # Bone-3 bottom (j==5) gets a lower threshold — hardest remaining endpoint
        thr = hard_threshold if j == 5 else threshold
        mv  = hm.max()
        if mv < thr:
            continue
        y, x = np.unravel_index(hm.argmax(), hm.shape)
        # Sub-pixel refinement
        if 1 < x < hm.shape[1]-1 and 1 < y < hm.shape[0]-1:
            dx = (hm[y, x+1] - hm[y, x-1]) / 2.0
            dy = (hm[y+1, x] - hm[y-1, x]) / 2.0
            x += dx * 0.25; y += dy * 0.25
        kps[j]  = [x / hm.shape[1], y / hm.shape[0]]   # normalised [0,1]
        conf[j] = mv

    # Anatomical constraint: bottom (odd j) must be below top (even j).
    # If top is found but bottom is missing, do a guided search below.
    for bone in range(3):          # bones 1-3
        j_top = bone * 2
        j_bot = bone * 2 + 1
        thr   = hard_threshold if j_bot == 5 else threshold
        if conf[j_top] >= threshold and conf[j_bot] < thr:
            top_y_pix = kps[j_top, 1] * hms.shape[1]
            hm   = hms[j_bot]
            ymin = max(0, int(top_y_pix))
            region = hm[ymin:, :]
            if region.size > 0 and region.max() >= hard_threshold * 0.7:
                ry, rx = np.unravel_index(region.argmax(), region.shape)
                abs_y  = ry + ymin
                kps[j_bot]  = [rx / hm.shape[1], abs_y / hm.shape[0]]
                conf[j_bot] = region.max()

    return kps, conf


def crop_to_image_coords(kps_crop, crop_box, orig_size):
    """
    kps_crop : (6,2) keypoints in [0,1] crop-relative space
    crop_box : (cx1, cy1, cx2, cy2) in original image pixels
    Returns  : (6,2) keypoints in original image pixels
    """
    cx1, cy1, cx2, cy2 = crop_box
    cw = cx2 - cx1; ch = cy2 - cy1
    kps_img = np.zeros_like(kps_crop)
    kps_img[:, 0] = kps_crop[:, 0] * cw + cx1
    kps_img[:, 1] = kps_crop[:, 1] * ch + cy1
    return kps_img

# =============================================================================
# BONE LENGTHS
# =============================================================================

def calculate_bone_lengths(all_kps, all_conf, conf_threshold):
    """
    all_kps  : (24,2) in original image pixels
    all_conf : (24,)
    Returns list of bone dicts.
    """
    bones = []
    for i in range(0, 24, 2):
        l1, l2 = ALL_KP_LABELS[i], ALL_KP_LABELS[i+1]
        bone_id = l1[:-1]
        c1, c2  = all_conf[i], all_conf[i+1]
        length  = (float(np.hypot(all_kps[i+1,0]-all_kps[i,0],
                                   all_kps[i+1,1]-all_kps[i,1]))
                   if c1 >= conf_threshold and c2 >= conf_threshold else None)
        bones.append({
            'bone_id': bone_id,
            'finger':  'Index' if bone_id[1] == '2' else 'Ring',
            'hand':    'Left'  if bone_id[0] == 'L' else 'Right',
            'segment': bone_id[2],
            'length_pixels':       length,
            'top_x':   float(all_kps[i,   0]), 'top_y':    float(all_kps[i,   1]),
            'bottom_x':float(all_kps[i+1, 0]), 'bottom_y': float(all_kps[i+1, 1]),
            'top_confidence':    float(c1),
            'bottom_confidence': float(c2),
        })
    return bones

# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize(orig_image, all_kps, all_conf, finger_masks_up,
              seg_boxes, bone_measurements, output_path, orig_size, cfg):
    img_arr = np.array(orig_image.convert('RGB'))
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    conf_thr = cfg['kp_threshold']

    # ---- Panel 1: keypoints + bones ----
    axes[0].imshow(img_arr, cmap='gray')
    axes[0].set_title('Cascade: Seg → Crop → Keypoints', fontsize=13, fontweight='bold')
    valid_count = 0
    for i, (kp, conf, label) in enumerate(zip(all_kps, all_conf, ALL_KP_LABELS)):
        thr = cfg['kp_threshold_hard'] if label.endswith('_') and label[2] == '3' else conf_thr
        if conf < thr:
            continue
        valid_count += 1
        fl    = label[:2]
        color = np.array(FINGER_COLORS[fl]) / 255.0
        axes[0].plot(kp[0], kp[1], 'x', color=color, markersize=4, markeredgewidth=0.8, alpha=0.9)
        if conf > 0.4:
            axes[0].text(kp[0]+8, kp[1]-8, label, color='yellow', fontsize=6,
                         fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.6))

    for i in range(0, 24, 2):
        l1, l2 = ALL_KP_LABELS[i], ALL_KP_LABELS[i+1]
        c1, c2 = all_conf[i], all_conf[i+1]
        if c1 >= conf_thr and c2 >= conf_thr:
            kp1, kp2 = all_kps[i], all_kps[i+1]
            color = np.array(FINGER_COLORS[l1[:2]]) / 255.0
            axes[0].plot([kp1[0], kp2[0]], [kp1[1], kp2[1]],
                         color=color, linewidth=0.8, alpha=0.7)
            length = np.hypot(kp2[0]-kp1[0], kp2[1]-kp1[1])
            axes[0].text((kp1[0]+kp2[0])/2, (kp1[1]+kp2[1])/2, f'{length:.0f}px',
                         color='white', fontsize=7, ha='center',
                         bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7))

    # Draw crop boxes used in stage 2
    for fl, box in seg_boxes.items():
        if box is None:
            continue
        cx1, cy1, cx2, cy2 = box
        color = np.array(FINGER_COLORS[fl]) / 255.0
        rect  = plt.Rectangle((cx1, cy1), cx2-cx1, cy2-cy1,
                               linewidth=1.5, edgecolor=color, facecolor='none',
                               linestyle='--', alpha=0.6)
        axes[0].add_patch(rect)

    axes[0].axis('off')
    axes[0].text(0.02, 0.98, f'Valid: {valid_count}/24',
                 transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # ---- Panel 2: segmentation masks ----
    axes[1].imshow(img_arr, cmap='gray')
    axes[1].set_title('Stage 1: Finger Segmentation Masks', fontsize=13, fontweight='bold')
    overlay = np.zeros((*img_arr.shape[:2], 4), dtype=np.float32)
    for ch, fl in enumerate(FINGER_LABELS):
        m = finger_masks_up[ch]
        color = np.array(FINGER_COLORS[fl]) / 255.0
        alpha_ch = (m > cfg['seg_threshold']).astype(np.float32) * 0.4
        for c_idx, cv_val in enumerate(color):
            overlay[:, :, c_idx] += alpha_ch * cv_val
        overlay[:, :, 3] = np.clip(overlay[:, :, 3] + alpha_ch, 0, 0.6)
    axes[1].imshow(overlay)
    patches = [mpatches.Patch(color=np.array(v)/255.0, label=k)
               for k, v in FINGER_COLORS.items()]
    axes[1].legend(handles=patches, loc='lower right', fontsize=9)
    axes[1].axis('off')

    # ---- Panel 3: summary ----
    axes[2].axis('off')
    axes[2].set_title('Measurement Summary', fontsize=13, fontweight='bold')
    valid_lengths = [b['length_pixels'] for b in bone_measurements
                     if b['length_pixels'] is not None]
    txt = f"Valid measurements: {len(valid_lengths)}/12\n\n"
    if valid_lengths:
        txt += (f"Overall:\n  Mean: {np.mean(valid_lengths):.1f}px\n"
                f"  Range: {np.min(valid_lengths):.1f} – {np.max(valid_lengths):.1f}px\n\n")
        for fl, fn, hand in [('L2','Index','Left'), ('L4','Ring','Left'),
                               ('R2','Index','Right'), ('R4','Ring','Right')]:
            fb = [b for b in bone_measurements
                  if b['bone_id'].startswith(fl) and b['length_pixels'] is not None]
            if fb:
                ll = [b['length_pixels'] for b in fb]
                txt += f"{hand} {fn}:\n  {len(ll)}/3 bones  Mean: {np.mean(ll):.1f}px\n\n"
    axes[2].text(0.05, 0.95, txt, transform=axes[2].transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("CASCADE PREDICTION  (Seg → Crop → Keypoints)")
    print("="*70 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load models ----
    for path in [CONFIG['seg_model_path'], CONFIG['kp_model_path']]:
        if not os.path.exists(path):
            print(f"ERROR: model not found: {path}")
            print("Run train_finger_seg.py and train_finger_kp.py first.")
            return

    seg_ck = torch.load(CONFIG['seg_model_path'], map_location=device)
    seg_model = FingerSegUNet(4, pretrained=False).to(device)
    seg_model.load_state_dict(seg_ck['model_state_dict'])
    seg_model.eval()
    print(f"Seg model loaded  (val_loss: {seg_ck.get('val_loss', 'N/A'):.4f})")

    kp_ck  = torch.load(CONFIG['kp_model_path'],  map_location=device)
    kp_model = FingerKpModel(6).to(device)
    kp_model.load_state_dict(kp_ck['model_state_dict'])
    kp_model.eval()
    print(f"KP model loaded   (val_loss: {kp_ck.get('val_loss', 'N/A'):.5f})\n")

    # ---- Output folders ----
    run_folder = os.path.join(CONFIG['output_folder'],
                              datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    vis_folder = os.path.join(run_folder, 'visualizations')
    os.makedirs(vis_folder, exist_ok=True)

    # ---- Image files ----
    exts = ['.jpg','.jpeg','.png','.tif','.tiff','.bmp']
    image_files = sorted(set(
        f for ext in exts
        for f in (list(Path(CONFIG['input_folder']).glob(f'*{ext}')) +
                  list(Path(CONFIG['input_folder']).glob(f'*{ext.upper()}')))
    ))
    if CONFIG.get('max_images'):
        image_files = image_files[:CONFIG['max_images']]
    print(f"Found {len(image_files)} images\n" + "-"*70)

    all_kp_rows, all_bone_rows = [], []

    for img_path in image_files:
        print(f"\n{img_path.name}")
        try:
            orig_image = Image.open(str(img_path)).convert('L')
            orig_size  = orig_image.size          # (width, height)
            orig_arr   = np.array(orig_image, dtype=np.uint8)

            # ================================================================
            # STAGE 1 — Finger Segmentation
            # ================================================================
            seg_input = preprocess_full(orig_image, CONFIG['image_size']).to(device)
            with torch.no_grad():
                seg_logits = seg_model(seg_input)                    # (1,4,512,512)
            seg_probs = torch.sigmoid(seg_logits)[0].cpu().numpy()   # (4,128,128)
            seg_probs = fix_finger_assignment(seg_probs, CONFIG['seg_threshold'])

            # Upsample masks to original image size for visualisation
            finger_masks_up = np.stack([
                cv2.resize(seg_probs[c], orig_size, interpolation=cv2.INTER_LINEAR)
                for c in range(4)
            ], axis=0)

            # ================================================================
            # STAGE 2 — Per-finger crop + keypoint detection
            # ================================================================
            all_kps  = np.zeros((24, 2), dtype=np.float32)
            all_conf = np.zeros(24, dtype=np.float32)
            seg_boxes = {}   # for visualisation

            for ch, finger_label in enumerate(FINGER_LABELS):
                # Find bbox of predicted finger mask (in model coordinate space)
                bbox_model = mask_to_bbox(seg_probs[ch], CONFIG['seg_threshold'])

                if bbox_model is None:
                    print(f"  [{finger_label}] No mask found — finger skipped")
                    seg_boxes[finger_label] = None
                    continue

                # Crop original image around that finger
                crop_arr, crop_box = make_square_crop(
                    orig_arr, bbox_model,
                    model_size=CONFIG['image_size'],
                    orig_size=orig_size,
                    pad_frac=CONFIG['pad_fraction'])

                if crop_arr is None or crop_arr.size == 0:
                    print(f"  [{finger_label}] Empty crop — skipped")
                    seg_boxes[finger_label] = None
                    continue

                seg_boxes[finger_label] = crop_box

                # Extract finger mask in crop coordinate space (same dilation as training)
                cx1, cy1, cx2, cy2 = crop_box
                finger_mask = finger_masks_up[ch, cy1:cy2, cx1:cx2]

                # Run keypoint model on masked crop
                crop_t = preprocess_crop(crop_arr, CONFIG['crop_size'],
                                         mask=finger_mask).to(device)
                kp_logits = predict_crop_tta(kp_model, crop_t, device)

                kp_names = FINGER_KP_NAMES[finger_label]
                kps_crop, conf_crop = extract_crop_keypoints(
                    kp_logits, kp_names,
                    CONFIG['kp_threshold'],
                    CONFIG['kp_threshold_hard'])

                # Transform to original image coordinates
                kps_img = crop_to_image_coords(kps_crop, crop_box, orig_size)

                # Write into global arrays (finger-specific slice)
                if finger_label == 'L2':
                    sl = slice(0,  6)
                elif finger_label == 'L4':
                    sl = slice(6,  12)
                elif finger_label == 'R2':
                    sl = slice(12, 18)
                else:  # R4
                    sl = slice(18, 24)

                all_kps[sl]  = kps_img
                all_conf[sl] = conf_crop

                n_valid = int((conf_crop >= CONFIG['kp_threshold']).sum())
                print(f"  [{finger_label}] crop={crop_box}  keypoints={n_valid}/6")

            # ================================================================
            # Results
            # ================================================================
            total_valid = int((all_conf >= CONFIG['kp_threshold']).sum())
            bone_measurements = calculate_bone_lengths(
                all_kps, all_conf, CONFIG['kp_threshold'])
            valid_bones = len([b for b in bone_measurements
                               if b['length_pixels'] is not None])
            print(f"  Total keypoints: {total_valid}/24   Bones: {valid_bones}/12")

            vis_path = os.path.join(vis_folder, f'{img_path.stem}_cascade.png')
            visualize(orig_image, all_kps, all_conf, finger_masks_up,
                      seg_boxes, bone_measurements, vis_path, orig_size, CONFIG)

            row = {'image': img_path.name, 'width': orig_size[0],
                   'height': orig_size[1], 'valid_keypoints': total_valid,
                   'valid_bones': valid_bones}
            for kp, conf, label in zip(all_kps, all_conf, ALL_KP_LABELS):
                row[f'{label}_x']    = float(kp[0])
                row[f'{label}_y']    = float(kp[1])
                row[f'{label}_conf'] = float(conf)
            all_kp_rows.append(row)
            for bone in bone_measurements:
                bone['image'] = img_path.name
                all_bone_rows.append(bone)

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}"); traceback.print_exc()

    if all_kp_rows:
        pd.DataFrame(all_kp_rows).to_csv(
            os.path.join(run_folder, 'keypoints.csv'), index=False)
        pd.DataFrame(all_bone_rows).to_csv(
            os.path.join(run_folder, 'bone_lengths.csv'), index=False)
        with open(os.path.join(run_folder, 'keypoints.json'), 'w') as f:
            json.dump(all_kp_rows, f, indent=2)
        with open(os.path.join(run_folder, 'bone_lengths.json'), 'w') as f:
            json.dump(all_bone_rows, f, indent=2)

    print("\n" + "="*70)
    print(f"Done. {len(all_kp_rows)} images processed.")
    print(f"Output: {run_folder}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
