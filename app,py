# app.py (root entrypoint for HF)
import os, pathlib

# ✅ fix Hugging Face HOME bug so Streamlit doesn't write to '/'
ROOT = pathlib.Path(__file__).parent
if os.environ.get("HOME", "/") in ("/", "", None):
    os.environ["HOME"] = str(ROOT)
os.environ.setdefault("XDG_CONFIG_HOME", os.environ["HOME"])
(pathlib.Path(os.environ["HOME"]) / ".streamlit").mkdir(exist_ok=True)

# 🔽 import your real app
import app.streamlit_app
