from fastapi import FastAPI, Body, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
from collections import defaultdict
import os
import json

from boxLabel import BoxLabel
import modelrun.scripts.tba.run_ocr as ocr

app = FastAPI()

# CORS 설정: 기존 포트(3000, 5173) 모두 허용
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 필드 정제/재구성 유틸리티
def refine_and_restructure_fields(raw_fields: list) -> dict:
    """
    OCR의 raw 결과를 받아 같은 라벨끼리 그룹핑하고 정렬하여 병합합니다.

    규칙:
    1. 같은 라벨끼리 그룹핑
    2. 주소 필드(adress 포함) 또는 조각이 여러개인 경우: y좌표 -> x좌표 순서로 정렬 후 병합
    3. 단일 조각 또는 비주소 필드: x좌표만으로 정렬 후 병합
    """
    grouped_by_label = defaultdict(list)

    for field in raw_fields or []:
        label = field.get("labels")
        text = field.get("value_text")
        box = field.get("value_box", {})

        # 텍스트가 없거나 박스가 없으면 건너뛰기 (라벨은 비어있을 수 있음)
        if not text or not box:
            print(f"필드 건너뛰기: 라벨='{label}', 텍스트='{text}', 박스={bool(box)}")
            continue

        ys = box.get("y") or []
        xs = box.get("x") or []
        center_y = sum(ys) / len(ys) if ys else 0
        start_x = min(xs) if xs else 0

        # 라벨이 비어있으면 고유한 키를 생성
        group_key = label if label else f"field_{field.get('id', len(grouped_by_label))}"

        grouped_by_label[group_key].append({
            "text": text,
            "center_y": center_y,
            "start_x": start_x,
        })

    final_fields: dict[str, str] = {}

    for label, fragments in grouped_by_label.items():
        if len(fragments) == 0:
            continue

        # 주소 필드이거나 조각이 여러개인 경우: y좌표 우선, 그 다음 x좌표로 정렬
        if "adress" in label.lower() or len(fragments) > 1:
            # y좌표(위에서 아래로), 같은 줄에서는 x좌표(왼쪽에서 오른쪽으로) 순서로 정렬
            fragments.sort(key=lambda f: (f["center_y"], f["start_x"]))
        else:
            # 단일 조각이거나 비주소 필드: x좌표만으로 정렬
            fragments.sort(key=lambda f: f["start_x"])

        # 텍스트 병합 (줄바꿈 없이 공백으로 연결)
        combined_text = " ".join([f['text'].strip() for f in fragments if f['text'].strip()])
        final_fields[label] = combined_text

    return final_fields


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.get("/api/v1/health")
def health_check():
    from datetime import datetime
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/document-types")
def get_document_types():
    """프론트엔드와 동일한 구조의 문서 종류 목록을 반환"""
    try:
        document_types_path = os.path.join(os.path.dirname(__file__), "document", "document_types.json")
        with open(document_types_path, 'r', encoding='utf-8') as f:
            document_types_data = json.load(f)

        return {"success": True, "data": document_types_data}
    except Exception as e:
        return {"success": False, "error": f"문서 종류 목록을 가져오는 중 오류가 발생했습니다: {str(e)}"}


@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon"}


async def _handle_ocr(selected_file: UploadFile, selected_doc_type: str):
    temp_filename = selected_file.filename if selected_file and selected_file.filename else ""
    try:
        print(f"OCR 시작: 파일={temp_filename}, 문서타입={selected_doc_type}")

        # 1) OCR 실행
        raw_ocr_data = await ocr.request_Ocr(selected_file, selected_doc_type)
        print(f"OCR 완료: 탐지된 필드 수={len(raw_ocr_data.get('fields', []))}")

        # 2) BoxLabel 처리
        processed_data = BoxLabel().get_data(raw_ocr_data)
        print(f"BoxLabel 처리 완료: 라벨링된 필드 수={len(processed_data.get('fields', []))}")

        # 3) 정제 함수를 호출하여 최종 결과 출력 형태로 만들기
        raw_fields = processed_data.get("fields", [])
        print(f"원본 필드 데이터: 필드 수={len(raw_fields)}")

        # 정제된 필드 생성 (라벨별 그룹핑 및 정렬)
        refined_fields = refine_and_restructure_fields(raw_fields)
        print(f"정제된 필드 생성: 필드 수={len(refined_fields)}")

        # 4) 최종 응답 구성 - 정제된 필드와 원본 필드 모두 제공
        final_response = {
            "metadata": processed_data.get("metadata"),
            "document_info": processed_data.get("document_info"),
            "refined_fields": refined_fields,  # 정제된 필드 (딕셔너리 형태)
            "raw_fields": raw_fields,  # 원본 필드 (배열 형태)
        }
        return final_response
    except FileNotFoundError as e:
        # 모델 파일 누락 등
        return {
            "status": "error",
            "message": "OCR 모델이 설정되지 않았습니다. 시스템 관리자에게 문의하세요.",
            "error_type": "model_not_found",
            "document_type": selected_doc_type,
            "filename": temp_filename,
            "ocr_results": [],
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "OCR 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
            "error_type": "processing_error",
            "document_type": selected_doc_type,
            "filename": temp_filename,
            "ocr_results": [],
        }


# 업로드 키명/폼 키명 호환 처리: filename|file, doc_type|document_type 모두 지원
@app.post("/insert")
async def board_insert(
    filename: Optional[UploadFile] = File(None, alias="filename"),
    file: Optional[UploadFile] = File(None, alias="file"),
    doc_type: Optional[str] = Form(None, alias="doc_type"),
    document_type: Optional[str] = Form(None, alias="document_type"),
):
    selected_file = filename or file
    selected_doc_type = doc_type or document_type
    if not selected_file or not selected_file.filename:
        raise HTTPException(status_code=400, detail="파일이 첨부되지 않았습니다.")
    if not selected_doc_type:
        raise HTTPException(status_code=400, detail="문서 종류(doc_type)가 선택되지 않았습니다.")
    return await _handle_ocr(selected_file, selected_doc_type)


@app.post("/api/v1/ocr/process")
async def process_ocr_api(
    filename: Optional[UploadFile] = File(None, alias="filename"),
    file: Optional[UploadFile] = File(None, alias="file"),
    doc_type: Optional[str] = Form(None, alias="doc_type"),
    document_type: Optional[str] = Form(None, alias="document_type"),
):
    selected_file = filename or file
    selected_doc_type = doc_type or document_type
    if not selected_file or not selected_file.filename:
        raise HTTPException(status_code=400, detail="파일이 첨부되지 않았습니다.")
    if not selected_doc_type:
        raise HTTPException(status_code=400, detail="문서 종류(doc_type)가 선택되지 않았습니다.")
    return await _handle_ocr(selected_file, selected_doc_type)


@app.get("/health")
async def health_check():
    """Railway 헬스체크 엔드포인트"""
    return {"status": "healthy", "service": "FinSight-OCR Backend"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)