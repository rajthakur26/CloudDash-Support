"""
Entry point for CloudDash Support API.
Run: python main.py
Or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import uvicorn
from api.app import app
from config.settings import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=(settings.app_env == "development"),
        log_level=settings.log_level.lower(),
    )
