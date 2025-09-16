import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import cv2
import os
import json
from tqdm import tqdm
import pyclipper
from shapely.geometry import Polygon
import wandb
from datetime import datetime
import albumentations as A
from pathlib import Path

# --- 1. 모델 아키텍처 (DBNet) ---
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 1-channel (grayscale) input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(self.maxpool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5

class DBNetFPN(nn.Module):
    def __init__(self, in_channels, inner_channels=256):
        super().__init__()
        self.in5 = nn.Conv2d(in_channels[3], inner_channels, 1)
        self.in4 = nn.Conv2d(in_channels[2], inner_channels, 1)
        self.in3 = nn.Conv2d(in_channels[1], inner_channels, 1)
        self.in2 = nn.Conv2d(in_channels[0], inner_channels, 1)
        self.out5 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1), nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1), nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1), nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1)
    def forward(self, features):
        c2, c3, c4, c5 = features
        p5 = self.in5(c5)
        p4 = self.in4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.in3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p2 = self.in2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        o2 = self.out2(p2)
        o3 = F.interpolate(self.out3(p3), size=o2.shape[2:], mode='nearest')
        o4 = F.interpolate(self.out4(p4), size=o2.shape[2:], mode='nearest')
        o5 = F.interpolate(self.out5(p5), size=o2.shape[2:], mode='nearest')
        fuse = torch.cat((o2, o3, o4, o5), 1)
        return fuse

class DBNetHead(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 3, padding=1)
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2)
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.ConvTranspose2d(in_channels // 4, out_channels, 2, 2)
    def forward(self, x):
        x = self.relu1(self.conv_bn1(self.conv1(x)))
        x = self.relu2(self.conv_bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class DBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.fpn = DBNetFPN(in_channels=[64, 128, 256, 512])
        self.head = DBNetHead(in_channels=256)
    def forward(self, x):
        features = self.backbone(x)
        fpn_out = self.fpn(features)
        maps = self.head(fpn_out)
        return maps

# --- 2. 손실 함수 (DBNetLoss) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        iflat, tflat = pred.contiguous().view(-1), target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth))

class DBNetLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0, ohem_ratio=3.0):
        super().__init__()
        self.alpha, self.beta, self.ohem_ratio = alpha, beta, ohem_ratio
        self.dice_loss = DiceLoss()
    def forward(self, preds, targets):
        prob_map, thresh_map, binary_map = preds[:, 0, :, :], preds[:, 1, :, :], preds[:, 2, :, :]
        gt_prob, gt_thresh, mask_prob, mask_thresh = targets['gt_prob'], targets['gt_thresh'], targets['mask_prob'], targets['mask_thresh']
        
        # Probability map loss with OHEM
        bce_loss = F.binary_cross_entropy_with_logits(prob_map, gt_prob, reduction='none')
        positive_mask = (gt_prob > 0).float()
        negative_mask = (gt_prob == 0).float()
        
        positive_loss = (bce_loss * positive_mask).sum() / (positive_mask.sum() + 1e-6)
        
        negative_loss_all = bce_loss * negative_mask
        num_negative = positive_mask.sum() * self.ohem_ratio
        num_negative = min(num_negative, negative_mask.sum())
        
        negative_loss, _ = torch.topk(negative_loss_all.view(-1), int(num_negative.item()))
        
        loss_prob = positive_loss + negative_loss.mean() + self.dice_loss(prob_map, gt_prob)
        
        # Threshold map loss
        loss_thresh = F.l1_loss(thresh_map * mask_thresh, gt_thresh * mask_thresh, reduction='mean')
        
        # Binarization map loss
        bce_binary_loss = F.binary_cross_entropy_with_logits(binary_map, gt_prob, reduction='none')
        loss_binary = (bce_binary_loss * positive_mask).mean() + self.dice_loss(binary_map, gt_prob)

        return self.alpha * loss_prob + self.beta * loss_thresh + loss_binary

# --- 3. 데이터셋 및 증강 ---
class DBNetDataset(Dataset):
    def __init__(self, image_paths, label_paths, is_train=True, img_size=960):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.is_train = is_train
        self.shrink_ratio = 0.4

        # 'keypoints'를 사용하도록 transform 파이프라인 수정
        if self.is_train:
            # 과적합 방지를 위한 더 강력한 데이터 증강 (버전 호환성 수정)
            self.transform = A.Compose([
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),  # value -> border_value
                A.Affine(scale=(0.9, 1.1), translate_percent=0.05, shear=(-8, 8), p=0.5),
                A.Perspective(scale=(0.05, 0.1), p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                A.GaussNoise(p=0.4),  # 경고 발생시키는 var_limit 제거
                A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                A.Resize(height=self.img_size, width=self.img_size, p=1.0),
                A.Normalize(mean=[0.5], std=[0.5]),
            ], keypoint_params=A.KeypointParams(format='xy'))
        else:
            self.transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size, p=1.0),
                A.Normalize(mean=[0.5], std=[0.5]),
            ], keypoint_params=A.KeypointParams(format='xy'))

    def __len__(self):
        return len(self.image_paths)

    def draw_threshold_map(self, polygon, canvas, shrink_ratio):
        poly = Polygon(polygon)
        if poly.area < 1: return
        try:
            distance = poly.area * (1 - shrink_ratio ** 2) / poly.length
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrunk_polygons = pco.Execute(-distance)
            if not shrunk_polygons: return
            cv2.fillPoly(canvas, [np.array(p).astype(np.int32) for p in shrunk_polygons], 0)
        except Exception:
            # pyclipper may fail on small or invalid polygons
            return

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        try:
            with open(img_path, 'rb') as f:
                img_buffer = np.frombuffer(f.read(), np.uint8)
            image = cv2.imdecode(img_buffer, cv2.IMREAD_GRAYSCALE)
            if image is None: raise IOError(f"Failed to decode image with imdecode: {img_path}")

            with open(label_path, 'r', encoding='utf-8') as f:
                label_json = json.load(f)
        except Exception as e:
            print(f"Warning: Skipping file due to error: {e}")
            return None

        polygons = []
        for bbox in label_json.get('bbox', []):
            x_coords, y_coords = bbox.get('x', []), bbox.get('y', [])
            if len(x_coords) != 4 or len(y_coords) != 4: continue
            
            poly = np.array([
                [x_coords[0], y_coords[0]], [x_coords[1], y_coords[1]],
                [x_coords[3], y_coords[3]], [x_coords[2], y_coords[2]]
            ], dtype=np.float32)

            if Polygon(poly).area < 1: continue
            polygons.append(poly)

        if self.transform:
            # Augmentation 라이브러리에 맞게 keypoints 형식으로 변환
            keypoints = [point for poly in polygons for point in poly]
            
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed['image']
            
            # 변환된 keypoints를 다시 polygon 형태로 복원
            transformed_polygons = []
            points_per_poly = 4
            transformed_points = transformed['keypoints']
            for i in range(0, len(transformed_points), points_per_poly):
                poly = np.array(transformed_points[i:i+points_per_poly], dtype=np.float32)
                if poly.shape[0] == points_per_poly:
                    transformed_polygons.append(poly)
            polygons = transformed_polygons

        h, w = image.shape
        gt_prob = np.zeros((h, w), dtype=np.float32)
        gt_thresh = np.zeros((h, w), dtype=np.float32)
        
        for poly in polygons:
            poly_int = poly.astype(np.int32)
            cv2.fillPoly(gt_prob, [poly_int], 1)
            cv2.fillPoly(gt_thresh, [poly_int], 1)
            self.draw_threshold_map(poly, gt_thresh, self.shrink_ratio)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        db_target = {
            "gt_prob": torch.from_numpy(gt_prob),
            "gt_thresh": torch.from_numpy(gt_thresh),
            "mask_prob": torch.ones_like(torch.from_numpy(gt_prob)),
            "mask_thresh": (torch.from_numpy(gt_prob) > 0).float()
        }
        return image_tensor, db_target

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None, None
    images, db_targets = zip(*batch)
    images = torch.stack(images, 0)
    
    db_target_batch = {}
    if db_targets:
        for key in db_targets[0].keys():
            # All images are resized to the same size, so no padding needed
            db_target_batch[key] = torch.stack([d[key] for d in db_targets], 0)
            
    return images, db_target_batch

# --- 4. 평가 함수 ---
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, db_targets in tqdm(val_loader, desc="Validating"):
            if images is None: continue
            images = images.to(device)
            for key in db_targets: db_targets[key] = db_targets[key].to(device)
            
            # Add autocast for mixed-precision evaluation
            with torch.cuda.amp.autocast():
                preds = model(images)
                loss = criterion(preds, db_targets)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
    # Handle case where val_loader is empty or all losses are NaN
    if len(val_loader) == 0:
        return 0.0
    return total_loss / len(val_loader)

# --- 5. 메인 학습 함수 ---
def main():
    # Configuration - A100 80GB Optimized
    DATA_DIR = Path("processed_data/detection_by_doctype_fixed")
    SAVE_DIR = Path("saved_models/detection_dbnet")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    config = {
        "img_size": 960,
        "batch_size": 28,  # 8 → 28 (A100 80GB 최적화)
        "epochs": 50,
        "lr": 1e-4,
        "weight_decay": 1e-3,  # 과적합 방지를 위한 Weight Decay 추가
        "grad_clip_norm": 5.0,
        "early_stopping_patience": 8,  # Patience 약간 증가 (강화된 Augmentation으로 인한 학습 안정화 시간 고려)
        "validation_per_epoch": 1,
    }

    run_name = f"DBNet_A100_Robust_{datetime.now().strftime('%y%m%d_%H%M%S')}"  # 이름 변경
    wandb.init(project="Finsight-OCR-Detection", name=run_name, config=config)
    config = wandb.config # Use wandb config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 데이터셋 준비 (Train/Val Split) ---
    print("Preparing datasets...")
    all_image_paths, all_label_paths = [], []
    from PIL import Image
    
    doc_folders = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    for folder in doc_folders:
        for split in ['train', 'val']:
            image_dir = folder / split / 'images'
            label_dir = folder / split / 'labels'
            for img_path in image_dir.glob("*.png"):
                # 손상된 이미지 사전 필터링
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                    label_path = label_dir / (img_path.stem + ".txt")
                    if label_path.exists():
                        # This is a dummy check; DBNet needs JSON, not TXT.
                        # The logic in the dataset will find the original JSON.
                        # This part needs to be fixed to find the original JSON path.
                        # For now, let's assume a parallel structure.
                        # This is a placeholder for a more robust file-finding logic.
                        # Let's assume the original JSONs are in `data/Training/02.라벨링데이터`
                        # and we can reconstruct the path.
                        # e.g., folder_01_금융_1.은행_1-1.신고서 -> TL_금융_1.은행_1-1.신고서
                        doc_type_str = "_".join(folder.name.split("_")[2:])
                        original_label_path = Path("data/Training/02.라벨링데이터") / f"TL_{doc_type_str}" / (img_path.stem + ".json")
                        
                        if original_label_path.exists():
                            all_image_paths.append(img_path)
                            all_label_paths.append(original_label_path)
                except Exception:
                    print(f"Skipping corrupted image: {img_path}")

    # Split all data into training and validation
    from sklearn.model_selection import train_test_split
    train_img, val_img, train_lbl, val_lbl = train_test_split(
        all_image_paths, all_label_paths, test_size=0.15, random_state=42
    )
    print(f"Training set size: {len(train_img)}")
    print(f"Validation set size: {len(val_img)}")

    train_dataset = DBNetDataset(train_img, train_lbl, is_train=True, img_size=config.img_size)
    val_dataset = DBNetDataset(val_img, val_lbl, is_train=False, img_size=config.img_size)

    # A100 80GB 최적화: num_workers 증가
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    # --- 모델 및 학습 설정 ---
    model = DBNet().to(device)
    # 과적합 방지를 위해 AdamW 옵티마이저 사용 및 weight_decay 적용
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.epochs, eta_min=1e-7)
    criterion = DBNetLoss().to(device)
    
    # A100 80GB 최적화: Mixed Precision Scaler 추가
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss, patience_counter = float('inf'), 0

    print("Starting DBNet model training with A100 80GB optimization...")
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for i, batch_data in enumerate(progress_bar):
            if batch_data is None: continue
            images, db_targets = batch_data
            if images is None: continue
            
            images = images.to(device)
            for key in db_targets: db_targets[key] = db_targets[key].to(device)
            
            optimizer.zero_grad()
            
            # A100 80GB 최적화: Mixed Precision으로 훈련
            with torch.cuda.amp.autocast():
                preds = model(images)
                loss = criterion(preds, db_targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at step {i}. Skipping batch.")
                continue
            
            # Mixed Precision Backward Pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{epoch_loss / (i+1):.4f}", best_loss=f"{best_val_loss:.4f}")
            wandb.log({"train_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]})

        # --- Validation Loop ---
        if (epoch + 1) % config.validation_per_epoch == 0:
            val_loss = evaluate(model, val_loader, criterion, device)
            wandb.log({"epoch": epoch + 1, "val_loss": val_loss})
            print(f"\nEpoch {epoch+1} - Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': dict(config)
                }, SAVE_DIR / "dbnet_a100_best.pth")
                print(f"🎉 New best model saved with Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= config.early_stopping_patience:
                print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs.")
                break
    
    print("DBNet A100 training finished.")
    wandb.finish()

if __name__ == '__main__':
    main()