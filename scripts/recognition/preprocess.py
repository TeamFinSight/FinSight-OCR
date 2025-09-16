import os
import json
import shutil
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_aihub_data(raw_data_dir, output_dir, val_split_ratio=0.1):
    """
    AI Hub 금융 문서 OCR 데이터셋을 EasyOCR 학습에 적합한 형식으로 전처리합니다.
    각 바운딩 박스 영역을 개별 이미지로 저장하고, 이에 대한 labels.csv 파일을 생성합니다.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_images_base_dir = os.path.join(output_dir, 'train_images')
    val_images_base_dir = os.path.join(output_dir, 'val_images')
    
    os.makedirs(train_images_base_dir, exist_ok=True)
    os.makedirs(val_images_base_dir, exist_ok=True)

    all_cropped_samples = [] # [(cropped_image_path, text_label)]

    # Define the main data splits (Training, Validation)
    data_splits = ['Training', 'Validation']

    logger.info(f"Processing raw data from: {raw_data_dir}")

    for data_split in data_splits:
        split_image_base_dir = os.path.join(raw_data_dir, data_split, '01.원천데이터')
        split_label_base_dir = os.path.join(raw_data_dir, data_split, '02.라벨링데이터')

        if not os.path.exists(split_image_base_dir) or not os.path.exists(split_label_base_dir):
            logger.warning(f"Skipping data split {data_split}: image or label base directory not found.")
            continue

        subfolders = [d for d in os.listdir(split_image_base_dir) if os.path.isdir(os.path.join(split_image_base_dir, d))]

        for subfolder_name in subfolders:
            cat_img_dir = os.path.join(split_image_base_dir, subfolder_name)

            if data_split == 'Training':
                expected_label_subfolder_name = subfolder_name.replace('TS_', 'TL_', 1)
            elif data_split == 'Validation':
                expected_label_subfolder_name = subfolder_name.replace('VS_', 'VL_', 1)
            else:
                expected_label_subfolder_name = subfolder_name

            cat_label_dir = os.path.join(split_label_base_dir, expected_label_subfolder_name)

            if not os.path.exists(cat_img_dir) or not os.path.exists(cat_label_dir):
                logger.warning(f"Skipping subfolder {subfolder_name} in {data_split}: corresponding image directory ({cat_img_dir}) or label directory ({cat_label_dir}) not found.")
                continue

            image_files = [f for f in os.listdir(cat_img_dir) if f.endswith('.png')]

            for img_file in tqdm(image_files, desc=f"Loading {data_split}/{subfolder_name} data"):
                try:
                    label_file = img_file.replace('.png', '.json')
                    img_path = os.path.join(cat_img_dir, img_file)
                    label_path = os.path.join(cat_label_dir, label_file)

                    if not os.path.exists(label_path):
                        logger.debug(f"Label file not found for {img_file}, skipping.")
                        continue
                    
                    original_image = Image.open(img_path).convert('RGB')
                    with open(label_path, 'r', encoding='utf-8') as f:
                        label_data = json.load(f)
                    
                    for bbox_idx, bbox_info in enumerate(label_data.get('bbox', [])):
                        data_type = bbox_info.get('data_type', -1)
                        text = bbox_info.get('data', '').strip()
                        
                        if data_type in [0, 3] and len(text) > 0: # Process text data
                            x_coords = bbox_info['x']
                            y_coords = bbox_info['y']
                            
                            x1, y1 = min(x_coords), min(y_coords)
                            x2, y2 = max(x_coords), max(y_coords)
                            
                            cropped_img = original_image.crop((x1, y1, x2, y2))
                            
                            # Ensure cropped image is not too small
                            if cropped_img.size[0] >= 8 and cropped_img.size[1] >= 8:
                                # Generate unique filename for cropped image
                                cropped_img_filename = f"{os.path.splitext(img_file)[0]}_{bbox_idx}.png"
                                all_cropped_samples.append((cropped_img, cropped_img_filename, text))

                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
                    continue
    
    if not all_cropped_samples:
        logger.error("No cropped samples found to process. Please check raw data directory and format.")
        return

    logger.info(f"Total {len(all_cropped_samples)} cropped samples loaded. Splitting into train/val sets...")
    
    train_samples, val_samples = train_test_split(all_cropped_samples, test_size=val_split_ratio, random_state=42)

    logger.info(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    def save_cropped_images_and_create_csv(samples, target_base_dir):
        records = []
        for cropped_img, filename, text in tqdm(samples, desc=f"Saving images and creating CSV for {os.path.basename(target_base_dir)}"):
            img_save_path = os.path.join(target_base_dir, filename)
            cropped_img.save(img_save_path)
            records.append({'filename': filename, 'words': text})
        
        df = pd.DataFrame(records)
        csv_save_path = os.path.join(target_base_dir, 'labels.csv')
        df.to_csv(csv_save_path, index=False, encoding='utf-8')
        logger.info(f"Created {csv_save_path} with {len(records)} records.")

    logger.info("Processing training data...")
    save_cropped_images_and_create_csv(train_samples, train_images_base_dir)
    
    logger.info("Processing validation data...")
    save_cropped_images_and_create_csv(val_samples, val_images_base_dir)

    logger.info("Data preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess AI Hub financial document OCR dataset for EasyOCR training.")
    parser.add_argument("--raw_data_dir", type=str, required=True,
                        help="Path to the raw AI Hub data directory (e.g., data/raw).")
    parser.add_argument("--output_dir", type=str, default="./processed_data",
                        help="Path to the output directory for processed data.")
    parser.add_argument("--val_split_ratio", type=float, default=0.1,
                        help="Ratio of validation data (e.g., 0.1 for 10%).")
    
    args = parser.parse_args()
    
    process_aihub_data(args.raw_data_dir, args.output_dir, args.val_split_ratio)