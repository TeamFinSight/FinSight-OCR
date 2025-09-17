#!/usr/bin/env python3

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2

def create_mask_from_json(json_path, image_size):
    """JSON 라벨에서 텍스트 마스크를 생성"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # bbox 정보가 있는지 확인
    if 'bbox' not in data or not data['bbox']:
        return mask  # 빈 마스크 반환
    
    # PIL을 사용해서 마스크 생성
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    
    for bbox in data['bbox']:
        if 'x' in bbox and 'y' in bbox:
            x_coords = bbox['x']
            y_coords = bbox['y']
            
            # 좌표가 4개인지 확인 (사각형)
            if len(x_coords) == 4 and len(y_coords) == 4:
                # 폴리곤 좌표 생성
                polygon = []
                for i in range(4):
                    polygon.append((x_coords[i], y_coords[i]))
                
                # 폴리곤 채우기
                draw.polygon(polygon, fill=255)
    
    return np.array(mask_img)

def create_yolo_labels_from_json(json_path, image_size):
    """JSON 라벨에서 YOLO 형식 라벨 생성"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    height, width = image_size
    labels = []
    
    if 'bbox' not in data or not data['bbox']:
        return labels
    
    for bbox in data['bbox']:
        if 'x' in bbox and 'y' in bbox:
            x_coords = bbox['x']
            y_coords = bbox['y']
            
            if len(x_coords) == 4 and len(y_coords) == 4:
                # 바운딩 박스 계산
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                
                # YOLO 형식으로 변환 (정규화된 중심점, 너비, 높이)
                center_x = (x_min + x_max) / 2 / width
                center_y = (y_min + y_max) / 2 / height
                bbox_width = (x_max - x_min) / width
                bbox_height = (y_max - y_min) / height
                
                # 클래스 ID는 0 (텍스트)
                labels.append(f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
    
    return labels

def fix_document_type_preprocessing():
    """문서 타입별 전처리 데이터 수정"""
    
    print("=== Fixing Document-Type-Specific Preprocessing ===")
    
    data_dir = Path("data/Training")
    images_dir = data_dir / "01.원천데이터"
    labels_dir = data_dir / "02.라벨링데이터"
    
    output_dir = Path("processed_data/detection_by_doctype_fixed")
    output_dir.mkdir(exist_ok=True)
    
    # 문서 타입별 폴더 찾기
    document_types = {}
    for label_folder in labels_dir.glob("TL_*"):
        if label_folder.is_dir():
            doc_type = label_folder.name.replace("TL_", "")
            corresponding_image_folder = images_dir / f"TS_{doc_type}"
            
            if corresponding_image_folder.exists():
                document_types[doc_type] = {
                    'images': corresponding_image_folder,
                    'labels': label_folder
                }
    
    print(f"Found {len(document_types)} document types")
    
    total_processed = 0
    total_valid = 0
    
    for idx, (doc_type, paths) in enumerate(document_types.items(), 1):
        print(f"\n=== Processing {idx}/{len(document_types)}: {doc_type} ===")
        
        # 출력 폴더 생성
        folder_name = f"folder_{idx:02d}_{doc_type}"
        doc_output_dir = output_dir / folder_name
        
        for split in ['train', 'val']:
            (doc_output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (doc_output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            (doc_output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
        
        # JSON 라벨 파일 목록
        json_files = list(paths['labels'].glob("*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        valid_count = 0
        processed_count = 0
        
        for json_file in tqdm(json_files, desc=f"Processing {doc_type}"):
            image_name = json_file.stem + ".png"
            image_path = paths['images'] / image_name
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            try:
                # 이미지 로드 및 크기 확인
                image = Image.open(image_path)
                image_size = (image.height, image.width)  # (H, W)
                
                # 마스크 생성
                mask = create_mask_from_json(json_file, image_size)
                
                # YOLO 라벨 생성
                yolo_labels = create_yolo_labels_from_json(json_file, image_size)
                
                # 텍스트가 있는 이미지만 처리
                if mask.sum() > 0 and len(yolo_labels) > 0:
                    # train/val 분할 (80:20)
                    split = 'train' if processed_count % 5 != 0 else 'val'
                    
                    # 파일 복사 및 저장
                    output_image_path = doc_output_dir / split / 'images' / image_name
                    output_mask_path = doc_output_dir / split / 'masks' / (json_file.stem + ".png")
                    output_label_path = doc_output_dir / split / 'labels' / (json_file.stem + ".txt")
                    
                    # 이미지 복사
                    shutil.copy2(image_path, output_image_path)
                    
                    # 마스크 저장
                    mask_img = Image.fromarray(mask)
                    mask_img.save(output_mask_path)
                    
                    # YOLO 라벨 저장
                    with open(output_label_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(yolo_labels))
                    
                    valid_count += 1
                else:
                    print(f"Warning: No text found in {image_name}")
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                continue
        
        print(f"Processed: {processed_count}, Valid: {valid_count}")
        total_processed += processed_count
        total_valid += valid_count
    
    print(f"\n=== Summary ===")
    print(f"Total processed: {total_processed}")
    print(f"Total valid: {total_valid}")
    print(f"Output directory: {output_dir}")
    
    # 각 폴더의 통계 출력
    print(f"\n=== Folder Statistics ===")
    for folder in sorted(output_dir.glob("folder_*")):
        train_images = len(list((folder / 'train' / 'images').glob("*.png")))
        val_images = len(list((folder / 'val' / 'images').glob("*.png")))
        train_labels = len(list((folder / 'train' / 'labels').glob("*.txt")))
        val_labels = len(list((folder / 'val' / 'labels').glob("*.txt")))
        
        print(f"{folder.name}: Train({train_images}img, {train_labels}lbl), Val({val_images}img, {val_labels}lbl)")

if __name__ == "__main__":
    fix_document_type_preprocessing()