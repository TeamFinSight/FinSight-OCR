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
    # "http://localhost:3000",
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


@app.post("/insert")
async def board_insert(
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
    json_data = await ocr.request_Ocr(filename, doc_type)
    box = BoxLabel()
    json_data = box.get_data(json_data)

    return json_data