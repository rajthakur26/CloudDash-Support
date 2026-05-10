import subprocess
import time

# Start FastAPI backend
subprocess.Popen([
    "uvicorn",
    "api.app:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
])

time.sleep(5)

# Start Streamlit frontend
subprocess.run([
    "streamlit",
    "run",
    "ui.py",
    "--server.port",
    "10000",
    "--server.address",
    "0.0.0.0"
])