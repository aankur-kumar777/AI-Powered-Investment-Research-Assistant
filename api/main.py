# api/main.py
from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="AI Investment Research Platform - API")
app.include_router(router, prefix="")

# optional simple root
@app.get("/")
def root():
    return {"message": "AI Investment Research Platform API. Try /health, /ingest, /query"}
