# OffPeak API (FastAPI)

## Setup
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Run (Git Bash)
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

## Check
http://127.0.0.1:8000/health
