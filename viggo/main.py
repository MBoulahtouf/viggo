# viggo/main.py
from fastapi import FastAPI
from viggo.api.v1.router import api_router
import mlflow
import wandb
from viggo.core.config import settings

app = FastAPI(
    title="Viggo API | Knowledge Explorer",
    description="An API to ask questions and explore knowledge graphs from documents.",
    version="1.0.0", # Let's call this version 1.0!
)

# Include the main router from our API module
app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Viggo API. Visit /docs for API documentation."}

# MLflow and WandB initialization
# mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
# mlflow.set_experiment(settings.wandb_project)
# try:
#     wandb.init(project=settings.wandb_project, job_type="api_session")
# except Exception as e:
#     print(f"WandB initialization failed: {e}. Continuing without WandB logging.")
