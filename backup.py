"""Run this file to make a zip backup of your project.
Usage:  python backup.py
It creates college_chatbot_backup_<date>.zip next to this file.
"""
import zipfile
from datetime import date
from pathlib import Path

PROJECT = Path(__file__).parent
SKIP = {"venv", ".venv", "__pycache__", ".git"}
zip_name = PROJECT / f"college_chatbot_backup_{date.today()}.zip"

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
    for f in PROJECT.rglob("*"):
        if f.is_file() and not (SKIP & set(f.parts)) and f.suffix != ".zip":
            z.write(f, f.relative_to(PROJECT))

print("Backup saved:", zip_name.name)