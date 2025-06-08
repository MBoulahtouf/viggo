# strider/main.py
from fastapi import FastAPI
from strider.api.v1.router import api_router

app = FastAPI(
    title="Strider API | Knowledge Explorer",
    description="An API to ask questions and explore knowledge graphs from documents.",
    version="1.0.0", # Let's call this version 1.0!
)

# Include the main router from our API module
app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Strider API. Visit /docs for API documentation."}
