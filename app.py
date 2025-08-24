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
import streamlit.web.bootstrap

app_path = str(ROOT / "app" / "streamlit_app.py")
streamlit.web.bootstrap.run(app_path, False, [], flag_options={})
