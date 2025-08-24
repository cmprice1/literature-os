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

def safe_escape_like(text):
    """Safely escape LIKE pattern wildcards"""
    if not isinstance(text, str):
        return text
    return text.replace('%', '\\%').replace('_', '\\_')

def query(c, tags=None, y0=None, y1=None, q=None):
    """Query papers with safe parameter handling"""
    try:
        where = []; params = []
        
        # Handle tags (ensure it's a list and not empty)
        if tags and isinstance(tags, (list, tuple)) and len(tags) > 0:
            where.append("id IN (SELECT paper_id FROM tags WHERE tag = ANY(%s))")
            params.append(list(tags))  # Convert to list to be safe
            
        # Handle years (ensure they're valid numbers)
        if y0 is not None and y1 is not None:
            try:
                y0, y1 = int(y0), int(y1)
                where.append("(year IS NOT NULL AND year BETWEEN %s AND %s)")
                params += [y0, y1]
            except (ValueError, TypeError):
                st.warning("Invalid year range, ignoring year filter")
                
        # Handle search query (escape special characters)
        if q and isinstance(q, str) and q.strip():
            q = safe_escape_like(q.strip())
            where.append("(title ILIKE %s OR COALESCE(abstract, '') ILIKE %s)")
            params += [f"%{q}%", f"%{q}%"]
        
        # Build and execute query
        sql = """
            SELECT 
                id,
                COALESCE(title, '(No title)') as title,
                COALESCE(journal, '(No journal)') as journal,
                year,
                COALESCE(doi, '') as doi,
                COALESCE(citation_count, 0) as citation_count
            FROM papers
        """
        
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY year DESC NULLS LAST, citation_count DESC NULLS LAST LIMIT 200"
        
        # Debug information in expandable section
        with st.expander("🔍 Query Debug Info", expanded=False):
            st.code(sql)
            st.json({"parameters": str(params) if params else "None"})
        
        # Execute query with error handling
        with c.cursor() as cur:
            cur.execute(sql, params or None)
            rows = cur.fetchall()
            
        # Convert to DataFrame with proper types
        df = pd.DataFrame(rows, columns=["id","Title","Journal","Year","DOI","Citations"])
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
        return df
        
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        st.code(f"SQL: {sql}\nParams: {params}")
        # Return empty DataFrame instead of raising
        return pd.DataFrame(columns=["id","Title","Journal","Year","DOI","Citations"])

# Main app with comprehensive error handling
try:
    st.title("Literature OS - MVP")
    
    # Database connection
    try:
        c = conn()
        st.success("Connected to database ✅")
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.stop()

    # Load metadata
    try:
        all_tags, years = load_tags_years(c)
        if not years:
            years = list(range(2010, 2026))  # Fallback year range
        if not all_tags:
            st.warning("No tags found in database")
            all_tags = []
    except Exception as e:
        st.error(f"Error loading metadata: {str(e)}")
        st.stop()

    # Filter section in a container for visual grouping
    with st.container():
        st.subheader("📋 Filter Papers")
        
        # Search box with clear instructions
        q = st.text_input(
            "Search titles & abstracts",
            help="Enter keywords to search in paper titles and abstracts"
        )

        # Year range with safe defaults
        col1, col2 = st.columns(2)
        with col1:
            yr_min = min(years) if years else 2010
            yr_max = max(years) if years else 2025
            yr0, yr1 = st.slider(
                "Year range",
                min_value=yr_min,
                max_value=yr_max,
                value=(max(2015, yr_min), yr_max),
                help="Filter papers by publication year"
            )

        # Tags with categorized options
        with col2:
            prefix_map = {
                "design:": "Research Design",
                "domain:": "Domain",
                "modality:": "Modality",
                "dx:": "Diagnosis"
            }
            
            # Group tags by prefix
            tag_groups = {prefix: [] for prefix in prefix_map.keys()}
            for tag in all_tags:
                for prefix in prefix_map.keys():
                    if tag.startswith(prefix):
                        tag_groups[prefix].append(tag)
                        break
            
            # Create multiselect for each group that has tags
            sel_tags = []
            for prefix, label in prefix_map.items():
                if tag_groups[prefix]:
                    group_tags = st.multiselect(
                        label,
                        options=sorted(tag_groups[prefix]),
                        help=f"Select {label.lower()} tags to filter by"
                    )
                    sel_tags.extend(group_tags)

    # Query and display results
    try:
        df = query(c, sel_tags, yr0, yr1, q)
        
        # Results section
        st.subheader(f"📚 Results ({len(df)} papers)")
        
        if len(df) == 0:
            st.info("No papers found matching your criteria. Try adjusting the filters.")
        else:
            # Add clickable DOI links
            df['DOI'] = df['DOI'].apply(lambda x: f'[{x}](https://doi.org/{x})' if x else '')
            
            # Display the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "DOI": st.column_config.LinkColumn("DOI"),
                    "Citations": st.column_config.NumberColumn("Citations", format="%d"),
                    "Year": st.column_config.NumberColumn("Year", format="%d")
                },
                hide_index=True
            )
            
            # Download button
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    "papers.csv",
                    "text/csv",
                    key='download-csv'
                )

    except Exception as e:
        st.error("Error loading results")
        with st.expander("Error details"):
            st.code(str(e))
        
except Exception as e:
    st.error("Application Error")
    with st.expander("Error details"):
        st.code(str(e))
