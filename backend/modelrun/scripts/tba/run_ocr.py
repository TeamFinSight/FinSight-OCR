import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import cv2
import numpy as np
import json
from datetime import datetime
import torchvision.models as models
import pyclipper
from shapely.geometry import Polygon

# --- 1. 모델 아키텍처 정의 (DBNet) ---
# train_final_detector.py와 완벽하게 동기화된 DBNet 구조
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
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

class RobustKoreanCRNN(nn.Module):
    """과적합 방지가 강화된 Korean handwriting recognition model"""
    def __init__(self, num_classes, lstm_hidden_size=384, lstm_layers=2, dropout_p=0.4):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b2', pretrained=False, 
                                        features_only=True, out_indices=[4], in_chans=1)
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 128, 1024)
            dummy_features = self.backbone(dummy_input)
            feature_dim = dummy_features[0].shape[1]
        self.korean_feature_enhance = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim//2, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(feature_dim//2), 
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(feature_dim//2, feature_dim//2, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(feature_dim//2), 
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        feature_dim = feature_dim // 2
        self.lstm1 = nn.LSTM(feature_dim, lstm_hidden_size, num_layers=1, 
                           bidirectional=True, batch_first=True, dropout=0)
        self.lstm2 = nn.LSTM(lstm_hidden_size * 2, lstm_hidden_size//2, num_layers=1,
                           bidirectional=True, batch_first=True, dropout=0)
        self.ln1 = nn.LayerNorm(lstm_hidden_size * 2)
        self.ln2 = nn.LayerNorm(lstm_hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size, 
                                             num_heads=4, dropout=dropout_p, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(lstm_hidden_size, lstm_hidden_size//2),
            nn.LayerNorm(lstm_hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.8),
            nn.Linear(lstm_hidden_size//2, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)[0]
        features = self.korean_feature_enhance(features)
        features = self.pool(features)
        features = features.squeeze(2).permute(0, 2, 1)
        lstm_out1, _ = self.lstm1(features)
        lstm_out1 = self.ln1(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.ln2(lstm_out2)
        if lstm_out1.size(-1) == lstm_out2.size(-1):
            lstm_combined = lstm_out1 + lstm_out2
        else:
            lstm_combined = lstm_out2
        attended_out, _ = self.attention(lstm_combined, lstm_combined, lstm_combined)
        attended_out = lstm_combined + attended_out
        output = self.classifier(attended_out)
        output = F.log_softmax(output, dim=2)
        return output.permute(1, 0, 2)

# --- 2. OCR 파이프라인 클래스 (DBNet 호환) ---
class OCR_Pipeline:
    def __init__(self, det_weights, rec_weights, char_map_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        with open(char_map_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f]
        self.char_map = ['[blank]'] + chars + ['[UNK]']
        num_classes = len(self.char_map)

        # --- 탐지 모델(DBNet) 로드 ---
        self.det_model = DBNet().to(self.device)
        det_checkpoint = torch.load(det_weights, map_location=self.device)
        self.det_model.load_state_dict(det_checkpoint['model_state_dict'])
        self.det_model.eval()
        print(f"탐지 모델(DBNet) 로드 완료: {det_weights}")

        # --- 인식 모델 로드 ---
        self.rec_model = RobustKoreanCRNN(num_classes=num_classes).to(self.device)
        rec_checkpoint = torch.load(rec_weights, map_location=self.device)
        self.rec_model.load_state_dict(rec_checkpoint['model_state_dict'])
        self.rec_model.eval()
        print("인식 모델 로드 완료.")

        # --- 전처리(Transform) 정의 ---
        self.det_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((960, 960)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.rec_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _crop_and_warp(self, image, polygon):
        # 입력 폴리곤을 4점 사각형으로 강제 변환
        rect = cv2.minAreaRect(polygon.astype(np.float32))
        ordered_pts = cv2.boxPoints(rect)
        
        # 면적 확인
        if cv2.contourArea(ordered_pts) < 1:
            return None

        # 정렬된 4개 꼭지점 확보
        ordered_pts = self._order_points_clockwise(ordered_pts)

        (tl, tr, br, bl) = ordered_pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        if maxWidth == 0 or maxHeight == 0: return None

        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def _ctc_decode(self, preds):
        # preds: (seq_len, num_classes) - log_softmax outputs
        preds_indices = preds.argmax(1) # Get the index of the most probable character for each timestep
        
        decoded_text = []
        confidence_score = 0.0
        last_char_index = -1
        
        for i, idx in enumerate(preds_indices):
            if idx != 0 and idx != last_char_index: # idx 0 is blank
                if idx < len(self.char_map):
                    decoded_text.append(self.char_map[idx])
                    # Add the log-probability of the predicted character
                    confidence_score += preds[i, idx].item()
            last_char_index = idx
            
        text = "".join(decoded_text)
        
        # Normalize confidence by length of the decoded text to avoid bias towards shorter words
        # If text is empty, confidence is 0.0
        if len(text) > 0:
            confidence_score /= len(text)
        else:
            confidence_score = -float('inf') # Or some other indicator for no text

        return text, confidence_score

    def predict(self, image_path, det_size=960, box_thresh=0.5, nms_thresh=0.2):
        try:
            with open(image_path, 'rb') as f:
                img_buffer = np.frombuffer(f.read(), np.uint8)
            original_image = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            if original_image is None: raise IOError(f"Failed to decode image with imdecode: {image_path}")
        except Exception as e:
            raise IOError(f"Cannot read image: {image_path}, Error: {e}")
        
        h_orig, w_orig, _ = original_image.shape
        image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        # 1. 텍스트 탐지 (DBNet)
        det_input = self.det_transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            det_output = self.det_model(det_input)
            prob_map = torch.sigmoid(det_output[:, 0, :, :]).squeeze(0).cpu().numpy()

        # 2. 후처리 및 폴리곤 추출 (단순화 및 안정화된 방식)
        binary_map = (prob_map > box_thresh).astype(np.uint8)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        initial_boxes = []
        scores = []
        angles = []
        for contour in contours:
            if cv2.contourArea(contour) < 3:
                continue
            
            # Pyclipper로 폴리곤 확장 (un-shrink)
            points = contour.reshape(-1, 2)
            try:
                poly = Polygon(points)
                distance = poly.area * 1.2 / poly.length # 여백 조절 (1.5 -> 1.2)
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                expanded_polygons = pco.Execute(distance)
                if not expanded_polygons: continue

                # 확장된 폴리곤을 4점 사각형으로 변환
                for exp_poly in expanded_polygons:
                    rect = cv2.minAreaRect(np.array(exp_poly))
                    box = cv2.boxPoints(rect)
                    initial_boxes.append(box)
                    scores.append(cv2.contourArea(box))
                    angles.append(rect[2])
            except Exception:
                continue

        print(f"Detected {len(initial_boxes)} initial boxes.")
        if not initial_boxes:
            return [], original_image

        # 3. NMS로 중복 박스 제거
        bounding_rects_for_nms = [cv2.boundingRect(box) for box in initial_boxes]
        indices = cv2.dnn.NMSBoxes(bounding_rects_for_nms, scores, score_threshold=0.1, nms_threshold=nms_thresh)

        if len(indices) == 0:
            final_polygons = []
            final_angles = []
        else:
            indices = indices.flatten()
            final_polygons = [initial_boxes[i] for i in indices]
            final_angles = [angles[i] for i in indices]
        
        if not final_polygons:
             print("No polygons left after NMS.")
             return [], original_image

        print(f"Kept {len(final_polygons)} polygons after NMS.")

        # 4. 텍스트 인식 (배치 처리)
        cropped_images, result_boxes, result_rotations = [], [], []
        for i, poly in enumerate(final_polygons):
            scaled_poly = (poly.astype(np.float32) * [w_orig / det_size, h_orig / det_size])
            
            warped_img = self._crop_and_warp(original_image, scaled_poly)
            if warped_img is None: continue

            target_height = 128
            max_width = 1024
            warped_pil = Image.fromarray(warped_img).convert("L")
            
            original_width, original_height = warped_pil.size
            if original_height == 0: continue

            aspect_ratio = original_width / original_height
            new_width = int(target_height * aspect_ratio)
            
            resized_pil = warped_pil.resize((new_width, target_height), Image.Resampling.LANCZOS)

            if new_width > max_width:
                final_img = resized_pil.crop((0, 0, max_width, target_height))
            else:
                final_img = Image.new("L", (max_width, target_height), 0)
                final_img.paste(resized_pil, (0, 0))
            
            cropped_images.append(self.rec_transform(final_img))
            result_boxes.append(scaled_poly.astype(np.int32))
            result_rotations.append(final_angles[i])

        if not cropped_images:
            return [], original_image

        rec_input = torch.stack(cropped_images).to(self.device)
        with torch.no_grad():
            rec_preds = self.rec_model(rec_input)
        
        results = []
        for i, pred in enumerate(rec_preds.permute(1, 0, 2)):
            text, confidence = self._ctc_decode(pred)
            results.append({"box": result_boxes[i].tolist(), "text": text, "rotation": result_rotations[i], "confidence": confidence})
        
        return results, original_image


    async def predict2(self, file, det_size=960, box_thresh=0.5, nms_thresh=0.2):
        content = await file.read()
        np_array = np.frombuffer(content, np.uint8)
        original_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if original_image is None:
            raise IOError("Failed to decode image")
        # 이제 img를 사용 가능

        h_orig, w_orig, _ = original_image.shape
        image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        # 1. 텍스트 탐지 (DBNet)
        det_input = self.det_transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            det_output = self.det_model(det_input)
            prob_map = torch.sigmoid(det_output[:, 0, :, :]).squeeze(0).cpu().numpy()

        # 2. 후처리 및 폴리곤 추출 (단순화 및 안정화된 방식)
        binary_map = (prob_map > box_thresh).astype(np.uint8)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        initial_boxes = []
        scores = []
        angles = []
        for contour in contours:
            if cv2.contourArea(contour) < 3:
                continue
            
            # Pyclipper로 폴리곤 확장 (un-shrink)
            points = contour.reshape(-1, 2)
            try:
                poly = Polygon(points)
                distance = poly.area * 1.2 / poly.length # 여백 조절 (1.5 -> 1.2)
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                expanded_polygons = pco.Execute(distance)
                if not expanded_polygons: continue

                # 확장된 폴리곤을 4점 사각형으로 변환
                for exp_poly in expanded_polygons:
                    rect = cv2.minAreaRect(np.array(exp_poly))
                    box = cv2.boxPoints(rect)
                    initial_boxes.append(box)
                    scores.append(cv2.contourArea(box))
                    angles.append(rect[2])
            except Exception:
                continue

        print(f"Detected {len(initial_boxes)} initial boxes.")
        if not initial_boxes:
            return [], original_image

        # 3. NMS로 중복 박스 제거
        bounding_rects_for_nms = [cv2.boundingRect(box) for box in initial_boxes]
        indices = cv2.dnn.NMSBoxes(bounding_rects_for_nms, scores, score_threshold=0.1, nms_threshold=nms_thresh)

        if len(indices) == 0:
            final_polygons = []
            final_angles = []
        else:
            indices = indices.flatten()
            final_polygons = [initial_boxes[i] for i in indices]
            final_angles = [angles[i] for i in indices]
        
        if not final_polygons:
             print("No polygons left after NMS.")
             return [], original_image

        print(f"Kept {len(final_polygons)} polygons after NMS.")

        # 4. 텍스트 인식 (배치 처리)
        cropped_images, result_boxes, result_rotations = [], [], []
        for i, poly in enumerate(final_polygons):
            scaled_poly = (poly.astype(np.float32) * [w_orig / det_size, h_orig / det_size])
            
            warped_img = self._crop_and_warp(original_image, scaled_poly)
            if warped_img is None: continue

            target_height = 128
            max_width = 1024
            warped_pil = Image.fromarray(warped_img).convert("L")
            
            original_width, original_height = warped_pil.size
            if original_height == 0: continue

            aspect_ratio = original_width / original_height
            new_width = int(target_height * aspect_ratio)
            
            resized_pil = warped_pil.resize((new_width, target_height), Image.Resampling.LANCZOS)

            if new_width > max_width:
                final_img = resized_pil.crop((0, 0, max_width, target_height))
            else:
                final_img = Image.new("L", (max_width, target_height), 0)
                final_img.paste(resized_pil, (0, 0))
            
            cropped_images.append(self.rec_transform(final_img))
            result_boxes.append(scaled_poly.astype(np.int32))
            result_rotations.append(final_angles[i])

        if not cropped_images:
            return [], original_image

        rec_input = torch.stack(cropped_images).to(self.device)
        with torch.no_grad():
            rec_preds = self.rec_model(rec_input)
        
        results = []
        for i, pred in enumerate(rec_preds.permute(1, 0, 2)):
            text, confidence = self._ctc_decode(pred)
            results.append({"box": result_boxes[i].tolist(), "text": text, "rotation": result_rotations[i], "confidence": confidence})
        
        return results, original_image

# --- 3. 커맨드라인 실행 부분 ---

async def request_Ocr(file, doc_type):

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="OCR 파이프라인 (DBNet 호환): 이미지에서 텍스트를 탐지하고 인식합니다.")
    # parser.add_argument("--det_weights", type=str, required=True, help="학습된 DBNet 탐지 모델(.pth)의 경로", default="saved_models/dbnet_a100_best.pth")
    # parser.add_argument("--rec_weights", type=str, required=True, help="학습된 인식 모델(.pth)의 경로", default="saved_models/robust_korean_recognition_best.pth")
    # parser.add_argument("--source", type=str, required=True, help="처리할 이미지 파일의 경로")
    # parser.add_argument("--char_map", type=str, default="configs/recognition/korean_char_map.txt", help="문자 맵 파일 경로")
    # parser.add_argument("--output_dir", type=str, default="output/pipeline", help="결과 이미지를 저장할 디렉토리")
    # parser.add_argument("--document_type", type=str, default="unknown", help="문서 종류 (JSON 결과에 포함)")
    # args = parser.parse_args()
    default_path = "modelrun/"
    det_weights = default_path+"saved_models/dbnet_a100_best.pth"
    rec_weights = default_path+"saved_models/robust_korean_recognition_best.pth"
    char_map = default_path+"configs/recognition/korean_char_map.txt"
    
    # 모델 파일 존재 여부 확인
    import os
    if not os.path.exists(det_weights):
        raise FileNotFoundError(f"탐지 모델 파일을 찾을 수 없습니다: {det_weights}")
    if not os.path.exists(rec_weights):
        raise FileNotFoundError(f"인식 모델 파일을 찾을 수 없습니다: {rec_weights}")
    if not os.path.exists(char_map):
        raise FileNotFoundError(f"문자 맵 파일을 찾을 수 없습니다: {char_map}")
    
    ocr_pipeline = OCR_Pipeline(det_weights, rec_weights, char_map)
    #===================

    print(f"\n--- OCR 처리 시작: args.source ---")
    results, original_image =  await ocr_pipeline.predict2(file)
    print(f"\n--- OCR 처리 완료 --- ({len(results)}개 텍스트 탐지)")

    output_dir = Path("output/pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_image = original_image.copy()

    from PIL import Image, ImageDraw, ImageFont
    vis_image_pil = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(vis_image_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for res in results:
        print(f"  - 좌표: {res['box']}, 예측 텍스트: \"{res['text']}\", 회전: {res.get('rotation', 0.0):.2f}")
        box_points = res['box']
        draw.polygon([tuple(p) for p in box_points], outline=(0, 255, 0), width=2)
        
        x, y = int(res['box'][0][0]), int(res['box'][0][1])
        text_bbox = draw.textbbox((x, y - 25), res['text'], font=font)
        draw.rectangle(text_bbox, fill="black")
        draw.text((x, y - 25), res['text'], fill="white", font=font)
    
    vis_image = cv2.cvtColor(np.array(vis_image_pil), cv2.COLOR_RGB2BGR)
    
    # save_path = output_dir / f"{Path(args.source).stem}_result.jpg"
    save_path = output_dir / f"{file.filename}_result.jpg"
    # Use robust image writing for Windows Unicode paths
    is_success, im_buf_arr = cv2.imencode(".jpg", vis_image)
    if is_success:
        im_buf_arr.tofile(str(save_path))
        print(f"\n결과 이미지가 {save_path} 에 저장되었습니다.")
    else:
        print("\n결과 이미지 저장에 실패했습니다.")

    # JSON 결과 저장 로직
    json_data = {
        "metadata": {
            "source_image": str(file.filename),
            "processed_at": datetime.now().isoformat(),
            "total_detections": len(results),
            "model_info": {
                "detection_model": "DBNet (ResNet18)",
                "recognition_model": "RobustKoreanCRNN (EfficientNet-B2 + BiLSTM + CTC)"
            }
        },
        "document_info": {
            "width": original_image.shape[1],
            "height": original_image.shape[0],
            "document_type": doc_type
        },
        "fields": [
            {
                "id": i + 1,
                "labels": "",  # Key-value 추출은 이 스크립트의 범위를 벗어남
                "rotation": res.get('rotation', 0.0),
                "value_text": res['text'],
                "confidence": res.get('confidence', -float('inf')),
                "value_box": {
                    "x": [p[0] for p in res['box']],
                    "y": [p[1] for p in res['box']],
                    "type": "polygon"
                }
            } for i, res in enumerate(results)
        ]
    }
    
    # json_path = output_dir / f"{Path(args.source).stem}_result.json"
    json_path = output_dir / f"{file.filename}_result.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"JSON 결과가 {json_path} 에 저장되었습니다.")
    return json_data

