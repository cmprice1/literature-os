import os
import pathlib
import streamlit as st
import pandas as pd
import psycopg
from psycopg_pool import ConnectionPool

# ──────────────────────────────────────────────────────────────────────────────
# Basic page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Literature OS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Configure Streamlit paths for HF Space
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if os.environ.get("HOME", "/") in ("/", "", None):
    os.environ["HOME"] = str(PROJECT_ROOT)
os.environ.setdefault("XDG_CONFIG_HOME", os.environ["HOME"])  # for Streamlit
os.makedirs(os.path.join(os.environ["HOME"], ".streamlit"), exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Database utilities (robust to idle timeouts / cold starts)
# ──────────────────────────────────────────────────────────────────────────────
DB_DSN = os.environ.get("DB_READER_DSN") or st.secrets.get("DB_READER_DSN", None)
if not DB_DSN:
    st.stop()

@st.cache_resource(show_spinner=False)
def get_pool() -> ConnectionPool:
    """Create a small connection pool. Using a pool avoids crashes when a single
    cached connection goes stale (common on Neon/managed PG + Spaces).
    """
    return ConnectionPool(
        conninfo=DB_DSN,
        min_size=1,
        max_size=5,
        max_lifetime=600,   # recycle connections periodically
        timeout=10,         # wait up to 10s for a free connection
        kwargs={"application_name": "literature-os-streamlit"},
    )


def run_sql(sql: str, params: list | tuple | None = None) -> list[tuple]:
    """Execute a read-only query safely via the pool and return rows.
    This pings the DB on each call so we don't reuse a dead connection.
    """
    pool = get_pool()
    # psycopg_pool connections are autocommit by default.
    with pool.connection() as conn:  # type: psycopg.Connection
        with conn.cursor() as cur:
            cur.execute("SELECT 1")  # health check; raises if connection is bad
            cur.execute(sql, params or [])
            return cur.fetchall()


# ──────────────────────────────────────────────────────────────────────────────
# Cached data fetchers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def get_metadata() -> tuple[list[str], list[int]]:
    """Load tags and available years from DB (cached)."""
    try:
        tags_rows = run_sql("SELECT DISTINCT tag FROM tags WHERE tag IS NOT NULL ORDER BY tag")
        years_rows = run_sql(
            """
            SELECT DISTINCT year
            FROM papers
            WHERE year IS NOT NULL AND year BETWEEN 1900 AND 2100
            ORDER BY year
            """
        )
        tags = [r[0] for r in tags_rows]
        years = [int(r[0]) for r in years_rows]
        return tags, years
    except Exception as e:
        st.warning(f"Failed to load metadata: {e}")
        # Sensible fallbacks
        return [], list(range(2010, 2026))


@st.cache_data(ttl=300, show_spinner=False)
def get_papers(tags: list[str] | None, year_start: int | None, year_end: int | None, search: str | None) -> pd.DataFrame:
    """Query papers with filters (cached by args, not by a connection object)."""
    sql = [
        """
        SELECT
            p.id,
            COALESCE(p.title, '(No title)') AS title,
            COALESCE(p.journal, '(No journal)') AS journal,
            p.year,
            COALESCE(p.doi, '') AS doi,
            COALESCE(p.citation_count, 0) AS citation_count
        FROM papers p
        WHERE 1=1
        """
    ]
    params: list = []

    if tags:
        # EXISTS is usually faster and avoids edge-cases with IN + ANY typing
        sql.append(" AND EXISTS (SELECT 1 FROM tags t WHERE t.paper_id = p.id AND t.tag = ANY(%s))")
        params.append(tags)

    if year_start is not None and year_end is not None:
        sql.append(" AND p.year BETWEEN %s AND %s")
        params.extend([int(year_start), int(year_end)])

    if search:
        q = f"%{search.strip()}%"
        sql.append(" AND (p.title ILIKE %s OR p.abstract ILIKE %s)")
        params.extend([q, q])

    sql.append(" ORDER BY p.year DESC NULLS LAST, p.citation_count DESC NULLS LAST LIMIT 200")

    try:
        rows = run_sql("".join(sql), params)
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame(columns=["ID", "Title", "Journal", "Year", "DOI", "Citations"])  # empty

    df = pd.DataFrame(rows, columns=["ID", "Title", "Journal", "Year", "DOI", "Citations"]) if rows else pd.DataFrame(columns=["ID", "Title", "Journal", "Year", "DOI", "Citations"])  

    # Clean types (safe for Streamlit front-end)
    if not df.empty:
        # Use pandas nullable integers to avoid JS formatter crashes on NaN
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
        # Build a plain-text DOI URL; avoid LinkColumn for now due to Streamlit bug with nulls
        def to_doi_url(x: str) -> str:
            x = (x or "").strip()
            return f"https://doi.org/{x}" if x else ""
        df["DOI"] = df["DOI"].map(to_doi_url)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

def render_results(df: pd.DataFrame) -> None:
    """Render results in a way that won't take the app down if Streamlit's
    dataframe frontend gets unhappy (e.g., with link columns / formatters)."""
    try:
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "ID": st.column_config.TextColumn("ID"),
                "Title": st.column_config.TextColumn("Title", width="medium"),
                "Journal": st.column_config.TextColumn("Journal", width="small"),
                # Avoid NumberColumn + %d which can crash on nulls; let Streamlit render natively
                "Year": st.column_config.TextColumn("Year"),
                "Citations": st.column_config.TextColumn("Citations"),
                # Use plain text for DOI to sidestep LinkColumn-related crashes; users can click the full URL
                "DOI": st.column_config.TextColumn("DOI"),
            },
            hide_index=True,
        )
    except Exception as e:
        st.warning(f"Standard table renderer failed ({e}). Falling back to safe static table.")
        st.table(df)

st.title("Literature OS")
st.caption("Last updated via GitHub → HF sync ✅")

# Make sure the pool spins up and the DB is reachable
try:
    _ = run_sql("SELECT 1")
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Load filter options
tags, years = get_metadata()

st.subheader("📋 Filter Papers")

# Determine slider bounds dynamically
if years:
    min_year, max_year = min(years), max(years)
else:
    min_year, max_year = 2010, 2025

# Default range: last 10 years or full range if shorter
default_start = max(min_year, max_year - 10)
default_range = (default_start, max_year)

col1, col2 = st.columns(2)
with col1:
    search = st.text_input(
        "Search titles & abstracts",
        help="Enter keywords to search in paper titles and abstracts",
    )
    year_range = st.slider(
        "Year range",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(default_range[0]), int(default_range[1])),
        help="Filter papers by publication year",
    )

with col2:
    selected_tags = st.multiselect(
        "Filter by tags",
        options=sorted(tags),
        help="Select tags to filter papers (OR logic)",
    )

# Query and display papers
papers = get_papers(
    tags=selected_tags,
    year_start=year_range[0],
    year_end=year_range[1],
    search=search,
)

st.subheader(f"📚 Results ({len(papers)} papers)")

if papers.empty:
    st.info("No papers found matching your criteria.")
else:
    render_results(papers)

    csv = papers.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download as CSV",
        data=csv,
        file_name="papers.csv",
        mime="text/csv",
        key="download-csv",
    )

with st.expander("ℹ️ Debug details"):
    st.write(
        "If you see intermittent crashes on filter changes, the original cause was a cached single PG connection going stale. This build uses a small connection pool and pings the DB on each query."
    )
    st.code(
        """
        - psycopg_pool.ConnectionPool(min_size=1, max_size=5, max_lifetime=600)
        - run_sql() does a SELECT 1 health check before each query
        - @st.cache_data functions no longer take a connection object, so cache keys are stable
        """,
        language="text",
    )
