# --- put these lines at the very top (before importing streamlit) ---
import os, pathlib

# Choose a writable directory for Streamlit config (repo root works fine)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # .../literature-os
# If HOME is '/' or empty, redirect it to the project root
if os.environ.get("HOME", "/") in ("/", "", None):
    os.environ["HOME"] = str(PROJECT_ROOT)

# Ensure config dir exists and point XDG there too
os.environ.setdefault("XDG_CONFIG_HOME", os.environ["HOME"])
CONFIG_DIR = os.path.join(os.environ["HOME"], ".streamlit")
os.makedirs(CONFIG_DIR, exist_ok=True)
# --------------------------------------------------------------------


import streamlit as st
import pandas as pd
import psycopg

# Set up your page config
st.set_page_config(page_title="Literature OS", layout="wide")

st.caption("Last updated via GitHub → HF sync ✅")

@st.cache_resource
def conn():
    return psycopg.connect(os.environ["DB_READER_DSN"])

def load_tags_years(c):
    with c.cursor() as cur:
        cur.execute("SELECT DISTINCT tag FROM tags ORDER BY tag")
        tags = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT year FROM papers WHERE year IS NOT NULL ORDER BY year DESC")
        years = [r[0] for r in cur.fetchall()]
    return tags, years

def query(c, tags, y0, y1, q):
    try:
        where = []; params = []
        if tags:
            where.append("id IN (SELECT paper_id FROM tags WHERE tag = ANY(%s))")
            params.append(tags)
        if y0 and y1:
            where.append("year BETWEEN %s AND %s"); params += [y0, y1]
        if q:
            where.append("(title ILIKE %s OR abstract ILIKE %s)"); params += [f"%{q}%", f"%{q}%"]
        
        sql = "SELECT id, title, journal, year, doi, citation_count FROM papers"
        if where: sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY year DESC NULLS LAST, citation_count DESC NULLS LAST LIMIT 200"
        
        # Debug information
        st.write("🔍 Debug Info (will remove in production):")
        with st.expander("Show SQL Query"):
            st.code(sql)
            st.json({"parameters": params if params else "None"})
        
        with c.cursor() as cur:
            cur.execute(sql, params or None)
            rows = cur.fetchall()
            
        return pd.DataFrame(rows, columns=["id","Title","Journal","Year","DOI","Citations"])
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        st.code(f"SQL: {sql}\nParams: {params}")
        raise e

try:
    st.title("Literature OS - MVP")
    
    # Database connection with error handling
    try:
        c = conn()
        st.success("Connected to database ✅")
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.stop()

    # Load tags and years with error handling
    try:
        all_tags, years = load_tags_years(c)
        if not all_tags or not years:
            st.warning("No data found in the database")
    except Exception as e:
        st.error(f"Error loading tags and years: {str(e)}")
        st.stop()

    # UI Elements with error handling
    try:
        sel_tags = st.multiselect("Tags", [t for t in all_tags if t.startswith(("design:","domain:","modality:","dx:"))])
        yr_min = min(years) if years else 2010
        yr_max = max(years) if years else 2025
        yr0, yr1 = st.slider("Year range", yr_min, yr_max, (max(2015, yr_min), yr_max))
        q = st.text_input("Search title or abstract")
    except Exception as e:
        st.error(f"Error in UI elements: {str(e)}")
        st.stop()

    # Query execution with error handling
    try:
        df = query(c, sel_tags, yr0, yr1, q)
        st.metric("Results", len(df))
        if len(df) == 0:
            st.info("No results found for your query")
        else:
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
