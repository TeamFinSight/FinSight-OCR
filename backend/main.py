from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fastapi import APIRouter, Depends 

from boxLabel import BoxLabel
import os
import json

import os, shutil
from typing import Optional
from fastapi import UploadFile, File, Form, HTTPException

import modelrun.scripts.tba.run_ocr as ocr

app = FastAPI()

origins = [
    # "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # "http://localhost:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

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
        # document_types.json 파일 읽기
        document_types_path = os.path.join(os.path.dirname(__file__), "document", "document_types.json")
        with open(document_types_path, 'r', encoding='utf-8') as f:
            document_types_data = json.load(f)

        return {
            "success": True,
            "data": document_types_data
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"문서 종류 목록을 가져오는 중 오류가 발생했습니다: {str(e)}"
        }


@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon"}


@app.post("/insert")
async def board_insert(
    filename:Optional[UploadFile] = File(None),
    doc_type:str = Form(...)
):
    return await process_ocr_request(filename, doc_type)

@app.post("/api/v1/ocr/process")
async def process_ocr_api(
    filename:Optional[UploadFile] = File(None),
    doc_type:str = Form(...)
):
    return await process_ocr_request(filename, doc_type)

async def process_ocr_request(
    filename:Optional[UploadFile] = File(None),
    doc_type:str = Form(...)
):
    if(doc_type == None):
        print("카테고리 비 선택")
        return "카데고리 선택"
    temp_filename = ""
    temp_image_url = ""

    if filename and filename.filename:
        temp_filename = filename.filename
        file_response = f'파일 {filename.filename}'
    else:
        file_response = "파일이 첨부되지 않았습니다."
    
    try:
        json_data = await ocr.request_Ocr(filename, doc_type)
        box = BoxLabel()
        json_data = box.get_data(json_data)
        return json_data
    except FileNotFoundError as e:
        print(f"OCR 모델 파일 오류: {str(e)}")
        # 모델 파일이 없는 경우 더미 데이터 반환
        return {
            "status": "error",
            "message": "OCR 모델이 설정되지 않았습니다. 시스템 관리자에게 문의하세요.",
            "error_type": "model_not_found",
            "document_type": doc_type,
            "filename": temp_filename,
            "ocr_results": []
        }
    except Exception as e:
        print(f"OCR 처리 중 오류 발생: {str(e)}")
        return {
            "status": "error",
            "message": "OCR 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
            "error_type": "processing_error",
            "document_type": doc_type,
            "filename": temp_filename,
            "ocr_results": []
        }