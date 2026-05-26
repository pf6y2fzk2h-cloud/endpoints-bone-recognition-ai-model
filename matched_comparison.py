#!/usr/bin/env python3.12
"""
Runs the AI model on the same training_images/ that were manually labelled,
then compares AI vs manual 2D:4D ratios row-by-row on the same images.

Output: matched_2d4d_comparison.csv  (load directly into Jamovi for ICC)
"""

import sys; sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image
import numpy as np
import cv2
import json
import csv
from pathlib import Path

CONFIG = {
    'annotations_file': '../labels.json',
    'images_dir':       '../training_images',
    'seg_model_path':   'trained_model/best_seg_model.pth',
    'kp_model_path':    'trained_model/best_kp_model.pth',
    'seg_threshold':    0.2,
    'pad_fraction':     0.10,
    'image_size':       256,
    'crop_size':        256,
    'output_file':      'matched_2d4d_comparison.csv',
}

FINGER_LABELS = ['L2', 'L4', 'R2', 'R4']
FINGER_KP_NAMES = {
    'L2': ['L21-','L21_','L22-','L22_','L23-','L23_'],
    'L4': ['L41-','L41_','L42-','L42_','L43-','L43_'],
    'R2': ['R21-','R21_','R22-','R22_','R23-','R23_'],
    'R4': ['R41-','R41_','R42-','R42_','R43-','R43_'],
}
FINGER_BONE_PAIRS = {
    'L2': [('L21-','L21_'),('L22-','L22_'),('L23-','L23_')],
    'L4': [('L41-','L41_'),('L42-','L42_'),('L43-','L43_')],
    'R2': [('R21-','R21_'),('R22-','R22_'),('R23-','R23_')],
    'R4': [('R41-','R41_'),('R42-','R42_'),('R43-','R43_')],
}

# =============================================================================
# MODELS
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
            bb.bn1, nn.ReLU(inplace=True))
        self.pool = bb.maxpool
        self.enc1 = bb.layer1; self.enc2 = bb.layer2
        self.enc3 = bb.layer3; self.enc4 = bb.layer4
        self.up4 = Up(512,256,256); self.up3 = Up(256,128,128)
        self.up2 = Up(128,64,64);   self.up1 = Up(64,64,32)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(32,32,2,stride=2), ConvBlock(32,32))
        self.head = nn.Conv2d(32, num_classes, 1)
    def forward(self, x):
        e0=self.enc0(x); e1=self.enc1(self.pool(e0))
        e2=self.enc2(e1); e3=self.enc3(e2); b=self.enc4(e3)
        d=self.up4(b,e3); d=self.up3(d,e2); d=self.up2(d,e1)
        d=self.up1(d,e0); d=self.up0(d)
        return self.head(d)

class FingerKpModel(nn.Module):
    def __init__(self, num_kp=6):
        super().__init__()
        self.enc1=ConvBlock(1,32);   self.pool1=nn.MaxPool2d(2)
        self.enc2=ConvBlock(32,64);  self.pool2=nn.MaxPool2d(2)
        self.enc3=ConvBlock(64,128); self.pool3=nn.MaxPool2d(2)
        self.bottle=ConvBlock(128,256)
        self.up3=nn.ConvTranspose2d(256,128,2,stride=2); self.dec3=ConvBlock(256,128)
        self.up2=nn.ConvTranspose2d(128,64,2,stride=2);  self.dec2=ConvBlock(128,64)
        self.up1=nn.ConvTranspose2d(64,32,2,stride=2);   self.dec1=ConvBlock(64,32)
        self.head=nn.Conv2d(32,num_kp,1)
    @staticmethod
    def _up(layer, x, ref):
        x = layer(x)
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool1(e1))
        e3=self.enc3(self.pool2(e2)); b=self.bottle(self.pool3(e3))
        d3=self.dec3(torch.cat([self._up(self.up3,b,e3),e3],dim=1))
        d2=self.dec2(torch.cat([self._up(self.up2,d3,e2),e2],dim=1))
        d1=self.dec1(torch.cat([self._up(self.up1,d2,e1),e1],dim=1))
        return self.head(d1)

# =============================================================================
# INFERENCE HELPERS
# =============================================================================

def apply_clahe(arr):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(arr)

def fix_finger_assignment(seg_probs, thr):
    def cx(ch):
        pts = np.where(seg_probs[ch] > thr)
        return float(pts[1].mean()) if len(pts[1]) > 0 else None
    l2_x, l4_x = cx(0), cx(1)
    if l2_x is not None and l4_x is not None and l2_x < l4_x:
        seg_probs[[0,1]] = seg_probs[[1,0]]
    r2_x, r4_x = cx(2), cx(3)
    if r2_x is not None and r4_x is not None and r2_x > r4_x:
        seg_probs[[2,3]] = seg_probs[[3,2]]
    return seg_probs

def predict_image(pil_img, seg_model, kp_model, device):
    arr = np.array(pil_img, dtype=np.uint8)
    orig_h, orig_w = arr.shape
    orig_size = (orig_w, orig_h)

    enhanced = apply_clahe(arr)
    sz = CONFIG['image_size']
    t = torch.from_numpy(cv2.resize(enhanced,(sz,sz)).astype(np.float32)/255.0)
    t = t.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(seg_model(t))[0].cpu().numpy()
    probs = fix_finger_assignment(probs, CONFIG['seg_threshold'])

    finger_masks_up = np.stack([
        cv2.resize(probs[c], orig_size, interpolation=cv2.INTER_LINEAR)
        for c in range(4)], axis=0)

    kp_coords = {}
    for ch, fl in enumerate(FINGER_LABELS):
        binary = (probs[ch] > CONFIG['seg_threshold']).astype(np.uint8)
        if binary.sum() == 0: continue
        n_lb, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        if n_lb <= 1: continue
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        bx1=int(stats[largest,cv2.CC_STAT_LEFT]); by1=int(stats[largest,cv2.CC_STAT_TOP])
        bx2=bx1+int(stats[largest,cv2.CC_STAT_WIDTH]); by2=by1+int(stats[largest,cv2.CC_STAT_HEIGHT])

        sx=orig_w/sz; sy=orig_h/sz
        x1,y1,x2,y2 = bx1*sx,by1*sy,bx2*sx,by2*sy
        pf=CONFIG['pad_fraction']
        x1-=(x2-x1)*pf; y1-=(y2-y1)*pf; x2+=(x2-x1)*pf; y2+=(y2-y1)*pf
        cx_=(x1+x2)/2; cy_=(y1+y2)/2; side=max(x2-x1,y2-y1)
        cx1=int(max(0,cx_-side/2)); cy1_=int(max(0,cy_-side/2))
        cx2=int(min(orig_w,cx_+side/2)); cy2=int(min(orig_h,cy_+side/2))
        if cx2<=cx1 or cy2<=cy1_: continue

        crop = arr[cy1_:cy2, cx1:cx2]
        if crop.size == 0: continue

        finger_mask = finger_masks_up[ch, cy1_:cy2, cx1:cx2]
        cs = CONFIG['crop_size']
        crop_e = apply_clahe(crop)
        crop_r = cv2.resize(crop_e, (cs,cs))
        m = cv2.resize(finger_mask,(cs,cs))
        kern=np.ones((9,9),np.uint8)
        m_dil=cv2.dilate((m>0.15).astype(np.uint8),kern,iterations=3).astype(np.float32)
        crop_r=np.clip(crop_r.astype(np.float32)*m_dil,0,255).astype(np.uint8)

        crop_t=torch.from_numpy(crop_r.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            hm=torch.sigmoid(kp_model(crop_t))[0].cpu().numpy()

        kp_labels = FINGER_KP_NAMES[fl]
        cw,ch_=cx2-cx1,cy2-cy1_
        for j,lbl in enumerate(kp_labels):
            y_,x_=np.unravel_index(hm[j].argmax(),hm[j].shape)
            nx=x_/hm[j].shape[1]*cw+cx1
            ny=y_/hm[j].shape[0]*ch_+cy1_
            kp_coords[lbl]=(nx,ny)

    return kp_coords

def finger_total_px(kp_coords, finger):
    total = 0.0
    all_detected = True
    for top_lbl, bot_lbl in FINGER_BONE_PAIRS[finger]:
        top = kp_coords.get(top_lbl)
        bot = kp_coords.get(bot_lbl)
        if top and bot:
            total += float(np.hypot(bot[0]-top[0], bot[1]-top[1]))
        else:
            all_detected = False
    return total, all_detected

# =============================================================================
# MAIN
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seg_ck = torch.load(CONFIG['seg_model_path'], map_location=device)
    seg_model = FingerSegUNet(4).to(device)
    seg_model.load_state_dict(seg_ck['model_state_dict']); seg_model.eval()

    kp_ck = torch.load(CONFIG['kp_model_path'], map_location=device)
    kp_model = FingerKpModel(6).to(device)
    kp_model.load_state_dict(kp_ck['model_state_dict']); kp_model.eval()
    print("Models loaded.\n")

    with open(CONFIG['annotations_file']) as f:
        all_data = json.load(f)

    # Only images with manual keypoint annotations
    annotated = [d for d in all_data if d.get('points')]
    print(f"Processing {len(annotated)} annotated images...\n")

    rows = []
    for img_data in annotated:
        img_name = img_data['name']
        img_path = Path(CONFIG['images_dir']) / img_name
        if not img_path.exists():
            print(f"  SKIP (not found): {img_name}")
            continue

        print(f"  {img_name} ...", end=' ')

        # ── Manual 2D:4D from annotations ────────────────────────────────────
        points = {p['label']: (p['x'], p['y']) for p in img_data.get('points', [])}
        manual = {}
        for finger in ['L2','L4','R2','R4']:
            total = 0.0
            for top_lbl, bot_lbl in FINGER_BONE_PAIRS[finger]:
                top = points.get(top_lbl)
                bot = points.get(bot_lbl)
                if top and bot:
                    total += float(np.hypot(bot[0]-top[0], bot[1]-top[1]))
            manual[finger] = total

        m_left  = manual['L2'] / manual['L4'] if manual['L4'] > 0 else None
        m_right = manual['R2'] / manual['R4'] if manual['R4'] > 0 else None

        # ── AI 2D:4D from model prediction ───────────────────────────────────
        try:
            pil_img = Image.open(img_path).convert('L')
            kp_coords = predict_image(pil_img, seg_model, kp_model, device)
            ai = {}
            for finger in ['L2','L4','R2','R4']:
                total, _ = finger_total_px(kp_coords, finger)
                ai[finger] = total
            ai_left  = ai['L2'] / ai['L4'] if ai['L4'] > 0 else None
            ai_right = ai['R2'] / ai['R4'] if ai['R4'] > 0 else None
        except Exception as e:
            print(f"ERROR: {e}")
            ai_left = ai_right = None

        print(f"manual L={m_left:.3f} R={m_right:.3f}  |  "
              f"AI L={ai_left:.3f} R={ai_right:.3f}" if m_left and ai_left else "")

        rows.append({
            'image':       img_name,
            'manual_left': round(m_left,  4) if m_left  is not None else '',
            'manual_right':round(m_right, 4) if m_right is not None else '',
            'ai_left':     round(ai_left,  4) if ai_left  is not None else '',
            'ai_right':    round(ai_right, 4) if ai_right is not None else '',
        })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(CONFIG['output_file'], 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image','manual_left','manual_right',
                                                'ai_left','ai_right'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} matched rows saved to: {CONFIG['output_file']}")
    print("Load this CSV into Jamovi for ICC analysis.")

if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
