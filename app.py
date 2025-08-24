# app.py (root entry point for Hugging Face Streamlit Space)
import os, pathlib

# Fix HOME bug on Hugging Face
ROOT = pathlib.Path(__file__).parent
if os.environ.get("HOME", "/") in ("/", "", None):
    os.environ["HOME"] = str(ROOT)
os.environ.setdefault("XDG_CONFIG_HOME", os.environ["HOME"])
(pathlib.Path(os.environ["HOME"]) / ".streamlit").mkdir(exist_ok=True)

# ✅ Import your actual Streamlit app (this will run its st.title etc.)
import app.streamlit_app
