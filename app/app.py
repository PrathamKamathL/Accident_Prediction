from fastapi import FastAPI
from fapp.schema import AccidentInput
from src.predict import predict

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hi"}
