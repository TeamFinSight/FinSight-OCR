import torch

# 1️⃣ 체크포인트 파일 경로
checkpoint_path = "C:/Users/Admin/Documents/GitHub/FinSight-OCR/saved_models/recognition/robust/robust_korean_recognition_best.pth"

# 2️⃣ 체크포인트 로드
checkpoint = torch.load(checkpoint_path, map_location='cpu')  # GPU 없이도 로드 가능

# 3️⃣ 저장된 key 확인
print("=== 저장된 key ===")
for key in checkpoint.keys():
    print(key)
print()

# 4️⃣ 에포크 정보
epoch = checkpoint.get('epoch', '정보 없음')
print(f"마지막 학습 에포크: {epoch}")

# 5️⃣ 최고 성능 정보
# CER, WER, Accuracy 등 저장된 metric 모두 확인
metrics = [k for k in checkpoint.keys() if 'best' in k.lower() or 'acc' in k.lower() or 'cer' in k.lower() or 'wer' in k.lower()]
if metrics:
    print("=== 최고 성능 ===")
    for metric in metrics:
        print(f"{metric}: {checkpoint[metric]}")
else:
    print("최고 성능 정보 없음")

# 6️⃣ 학습 설정(config) 확인
config = checkpoint.get('config', None)
if config:
    print("\n=== 학습 설정 (config) ===")
    if isinstance(config, dict):
        for k, v in config.items():
            print(f"{k}: {v}")
    else:
        print(config)
else:
    print("\n학습 설정 정보 없음")

# 7️⃣ 모델 구조 확인 (state_dict)
state_dict = checkpoint.get('model_state_dict', None)
if state_dict:
    print("\n=== 모델 레이어 및 파라미터 shape ===")
    for name, param in state_dict.items():
        print(f"{name}: {tuple(param.shape)}")
else:
    print("\n모델 state_dict 정보 없음")
