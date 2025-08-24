# app.py
import os, pathlib

# Fix Hugging Face HOME issue
ROOT = pathlib.Path(__file__).parent
if os.environ.get("HOME", "/") in ("/", "", None):
    os.environ["HOME"] = str(ROOT)
os.environ.setdefault("XDG_CONFIG_HOME", os.environ["HOME"])
(pathlib.Path(os.environ["HOME"]) / ".streamlit").mkdir(exist_ok=True)

# ✅ Make sure HF sees Streamlit
import streamlit as st

st.title("📚 Literature OS")
st.write("Hello from Hugging Face! If you see this, the app is running.")

# Load your real app
import app.streamlit_app
