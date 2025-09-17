from fastapi import FastAPI, Body, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
from collections import defaultdict

# --- 기존 import 구문 ---
from boxLabel import BoxLabel
import modelrun.scripts.tba.run_ocr as ocr
# --- (필요하다면 다른 import 구문 추가) ---

app = FastAPI()

# --- CORS 설정 (기존과 동일) ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# ✨ 1. 여기에 "사혼의 구슬 조각"을 모으는 함수를 추가합니다. ✨
# ==============================================================================
def refine_and_restructure_fields(raw_fields: list) -> dict:
    """
    OCR의 raw 결과를 받아 중복 필드를 병합하고,
    특히 '주소' 필드는 다중 라인을 고려하여 정교하게 재조립합니다.
    """
    grouped_by_label = defaultdict(list)
    for field in raw_fields:
        label = field.get("labels")
        text = field.get("value_text")
        box = field.get("value_box", {})
        
        if label and text and box:
            # y좌표 중심점(줄 그룹핑 기준), x좌표 시작점(줄 내 정렬 기준)을 계산
            center_y = sum(box.get("y", [])) / len(box.get("y", [])) if box.get("y") else 0
            start_x = min(box.get("x", [])) if box.get("x") else 0
            grouped_by_label[label].append({'text': text, 'center_y': center_y, 'start_x': start_x})

    final_fields = {}
    for label, fragments in grouped_by_label.items():
        # 'adress'를 포함하는 라벨이고, 조각이 2개 이상일 때 특별 처리
        if 'adress' in label and len(fragments) > 1:
            # 1. y좌표 기준 정렬 (위 -> 아래 순서)
            fragments.sort(key=lambda f: f['center_y'])
            
            # 2. x좌표 기준 정렬 후 텍스트 병합 (왼쪽 -> 오른쪽 순서)
            # (간단한 버전: 줄바꿈 없이 y,x 순서대로 모두 합칩니다.)
            fragments.sort(key=lambda f: (f['center_y'], f['start_x']))
            combined_text = " ".join([f['text'] for f in fragments])
            final_fields[label] = combined_text
            
        else:
            # 주소가 아니거나 조각이 1개인 다른 필드들은 x좌표 기준으로만 정렬
            fragments.sort(key=lambda f: f['start_x'])
            final_fields[label] = " ".join([f['text'] for f in fragments])
            
    return final_fields
# ==============================================================================

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/insert")
async def board_insert(
    filename: Optional[UploadFile] = File(None),
    doc_type: str = Form(...)
):
    if not filename or not filename.filename:
        raise HTTPException(status_code=400, detail="파일이 첨부되지 않았습니다.")
    if not doc_type:
        raise HTTPException(status_code=400, detail="문서 종류(doc_type)가 선택되지 않았습니다.")

    # 1. OCR 파이프라인 실행
    raw_ocr_data = await ocr.request_Ocr(filename, doc_type)
    
    # 2. boxLabel 처리 (기존 로직 유지)
    processed_data = BoxLabel().get_data(raw_ocr_data)
    
    # ==============================================================================
    # ✨ 2. 바로 이 부분에서 정제 함수를 호출하여 최종 결과를 만듭니다. ✨
    # ==============================================================================
    
    # processed_data에서 'fields' 리스트를 가져옵니다.
    raw_fields = processed_data.get("fields", [])
    
    # 정제 함수를 호출하여 깔끔한 Key-Value 딕셔너리를 얻습니다.
    final_structured_fields = refine_and_restructure_fields(raw_fields)

    # 최종 응답 데이터 구성
    final_response = {
        "metadata": processed_data.get("metadata"),
        "document_info": processed_data.get("document_info"),
        "fields": final_structured_fields  # ✨ 정제된 결과로 교체! ✨
    }

    return final_response