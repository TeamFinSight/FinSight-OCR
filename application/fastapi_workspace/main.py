from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
