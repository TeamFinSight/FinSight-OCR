#!/usr/bin/env python3
"""
Enhanced Korean Handwriting Recognition with Aspect-Ratio-Preserving Padding
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import wandb
import random
from torch.optim import lr_scheduler
import Levenshtein
import kornia.augmentation as K
import math
import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import re

# GPU 메모리 최적화
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

class EnhancedKoreanCRNN(nn.Module):
    """Enhanced Korean handwriting recognition model"""
    def __init__(self, num_classes, lstm_hidden_size=512, lstm_layers=3, dropout_p=0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, 
                                        features_only=True, out_indices=[4], in_chans=1)
        with torch.no_grad():
            # Input size now reflects max_width
            dummy_input = torch.randn(1, 1, 128, 1024)
            dummy_features = self.backbone(dummy_input)
            feature_dim = dummy_features[0].shape[1]
            print(f"Feature dimension: {feature_dim}")
        
        self.korean_feature_enhance = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, (1, 7), padding=(0, 3), groups=feature_dim//4),
            nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, (7, 1), padding=(3, 0), groups=feature_dim//4),
            nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm1 = nn.LSTM(feature_dim, lstm_hidden_size, num_layers=1, 
                           bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_hidden_size * 2, lstm_hidden_size, num_layers=1,
                           bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(lstm_hidden_size * 2, lstm_hidden_size, num_layers=1,
                           bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size * 2, 
                                             num_heads=8, dropout=dropout_p, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(lstm_hidden_size, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)[0]
        features = self.korean_feature_enhance(features)
        features = self.pool(features)
        features = features.squeeze(2).permute(0, 2, 1)
        lstm_out1, _ = self.lstm1(features)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out3, _ = self.lstm3(lstm_out2)
        lstm_out = lstm_out1 + lstm_out2 + lstm_out3
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended_out = lstm_out + attended_out
        output = self.classifier(attended_out)
        output = F.log_softmax(output, dim=2)
        return output.permute(1, 0, 2)

class RecognitionDataset(Dataset):
    """ Aspect-ratio-preserving dataset for OCR """
    def __init__(self, image_dir, label_csv_path, char_to_int, img_h=128, img_w=1024, is_training=False):
        self.image_dir = Path(image_dir)
        self.char_to_int = char_to_int
        self.is_training = is_training
        self.img_h = img_h
        self.img_w = img_w
        
        self.image_paths, self.labels = self._load_data(label_csv_path)
        
        self.base_transform = transforms.ToTensor()
        
        if is_training:
            self.augment = K.AugmentationSequential(
                K.RandomAffine(degrees=(-3, 3), translate=(0.02, 0.02), scale=(0.95, 1.05), p=0.7),
                K.RandomPerspective(distortion_scale=0.1, p=0.4),
                K.ColorJitter(brightness=0.2, contrast=0.2, p=0.4),
                K.RandomGaussianNoise(mean=0, std=0.02, p=0.3),
                data_keys=["input"],
                same_on_batch=False
            )
        else:
            self.augment = None

    def _load_data(self, label_csv_path):
        print(f"Loading labels: {label_csv_path}")
        df = pd.read_csv(label_csv_path)
        
        valid_paths, valid_labels = [], []
        print("Verifying dataset...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying"):
            img_name, text = row.iloc[0], str(row.iloc[1]).strip()
            
            if not (0 < len(text) <= 50): continue
            
            valid_chars = all(c in self.char_to_int or c == ' ' for c in text)
            if not valid_chars: continue
                
            full_path = self.image_dir / img_name
            if full_path.exists():
                valid_paths.append(full_path)
                valid_labels.append(text)
        
        print(f"Found {len(valid_paths)} valid image-label pairs.")
        return valid_paths, valid_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, text = self.image_paths[idx], self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("L")
            
            # --- Aspect-ratio-preserving resize and padding ---
            original_width, original_height = image.size
            if original_height == 0: raise ValueError("Image height is 0")

            aspect_ratio = original_width / original_height
            new_width = int(self.img_h * aspect_ratio)
            
            resized_img = image.resize((new_width, self.img_h), Image.Resampling.LANCZOS)
            
            padded_img = Image.new("L", (self.img_w, self.img_h), 0)
            padded_img.paste(resized_img, (0, 0))
            
            image_tensor = self.base_transform(padded_img)
            image_tensor = (image_tensor - 0.5) / 0.5
            
            encoded_text = [self.char_to_int.get(char, self.char_to_int['[UNK]']) for char in text]
            
            return image_tensor, torch.tensor(encoded_text, dtype=torch.long), text
            
        except Exception as e:
            print(f"Warning: Skipping {img_path} due to error: {e}")
            # Return a valid dummy item to avoid crashing the loader
            dummy_img = torch.zeros((1, self.img_h, self.img_w))
            dummy_text = torch.tensor([self.char_to_int['[UNK]']], dtype=torch.long)
            return dummy_img, dummy_text, "[ERROR]"

def collate_fn(batch):
    images, encoded_labels, text_labels = zip(*batch)
    images = torch.stack(images, 0)
    label_lengths = torch.tensor([len(lbl) for lbl in encoded_labels], dtype=torch.long)
    labels_flat = torch.cat(encoded_labels, dim=0)
    return images, labels_flat, label_lengths, text_labels

def ctc_decode(preds, char_map):
    preds_indices = preds.argmax(1).cpu().numpy()
    decoded_text, last_char_index = [], -1
    for idx in preds_indices:
        if idx != 0 and idx != last_char_index:
            if 0 < idx < len(char_map):
                decoded_text.append(char_map[idx])
        last_char_index = idx
    return "".join(decoded_text)

def calculate_cer(ground_truth, prediction):
    if len(ground_truth) == 0:
        return 1.0 if len(prediction) > 0 else 0.0
    return Levenshtein.distance(ground_truth, prediction) / len(ground_truth)

def main(args):
    wandb.init(project=args.wandb_project, name=f"rec-padded-{datetime.now().strftime('%Y%m%d_%H%M')}", config=args)
    config = wandb.config
    device = torch.device(config.device)
    
    print("=== Korean Handwriting Recognition (Padded) ===")

    with open(config.char_map, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f]
    char_map = ['[blank]'] + chars + ['[UNK]']
    char_to_int = {char: i for i, char in enumerate(char_map)}
    num_classes = len(char_map)
    print(f"Character map size: {num_classes}")

    train_dataset = RecognitionDataset(config.train_data_dir, Path(config.train_data_dir) / "labels.csv", char_to_int, is_training=True, img_h=config.img_h, img_w=config.img_w)
    val_dataset = RecognitionDataset(config.val_data_dir, Path(config.val_data_dir) / "labels.csv", char_to_int, is_training=False, img_h=config.img_h, img_w=config.img_w)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = EnhancedKoreanCRNN(num_classes, lstm_hidden_size=config.lstm_hidden_size, lstm_layers=config.lstm_layers, dropout_p=config.dropout_p).to(device)
    
    criterion = nn.CTCLoss(blank=char_to_int['[blank]'], zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=len(train_loader), epochs=config.epochs, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    
    best_cer = float('inf')
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        
        for images, labels, label_lengths, _ in pbar:
            images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
            
            if train_dataset.augment:
                images = train_dataset.augment(images)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(images)
                input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long, device=device)
                loss = criterion(preds, labels, input_lengths, label_lengths)
            
            if torch.isinf(loss) or torch.isnan(loss): continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss, total_cer, total_samples = 0.0, 0.0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]")
        
        with torch.no_grad():
            for images, labels, label_lengths, text_labels in val_pbar:
                images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
                
                with torch.cuda.amp.autocast():
                    preds = model(images)
                    input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long, device=device)
                    loss = criterion(preds, labels, input_lengths, label_lengths)
                
                if not (torch.isinf(loss) or torch.isnan(loss)):
                    val_loss += loss.item()
                
                for i in range(preds.size(1)):
                    pred_sample = preds[:, i, :]
                    decoded = ctc_decode(pred_sample, char_map)
                    total_cer += calculate_cer(text_labels[i], decoded)
                total_samples += len(text_labels)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_cer = total_cer / total_samples
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val CER: {avg_cer:.4f}")
        
        wandb.log({
            "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "val_cer": avg_cer, "learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch + 1
        })
        
        if avg_cer < best_cer:
            best_cer = avg_cer
            save_path = Path(config.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cer': best_cer,
                'config': dict(config)
            }, save_path / "korean_recognition_best_padded.pth")
            print(f"🎉 New best model saved with CER: {best_cer:.4f}")

    print(f"\nTraining complete. Best CER: {best_cer:.4f}")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced Korean Handwriting Recognition with Padding")
    
    parser.add_argument("--train_data_dir", type=str, default="processed_data/recognition/train_images")
    parser.add_argument("--val_data_dir", type=str, default="processed_data/recognition/val_images")
    parser.add_argument("--char_map", type=str, default="configs/recognition/korean_char_map.txt")
    parser.add_argument("--save_dir", type=str, default="saved_models/recognition/enhanced_padded")
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--lr", type=float, default=4.6875e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    parser.add_argument("--img_h", type=int, default=128)
    parser.add_argument("--img_w", type=int, default=1024) # Increased max width
    
    parser.add_argument("--lstm_hidden_size", type=int, default=512)
    parser.add_argument("--lstm_layers", type=int, default=3)
    parser.add_argument("--dropout_p", type=float, default=0.3)
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_project", type=str, default="Korean-Handwriting-Recognition-Padded")
    
    args = parser.parse_args()
    main(args)