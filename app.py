import os
import pandas as pd
import gradio as gr
from psycopg_pool import ConnectionPool

# --- Lazy pool (don't touch DB at import time) ---
_POOL = None
def get_pool() -> ConnectionPool:
    global _POOL
    if _POOL is None:
        dsn = os.environ.get("DB_READER_DSN")
        if not dsn:
            raise RuntimeError("Missing DB_READER_DSN secret.")
        _POOL = ConnectionPool(
            conninfo=dsn, min_size=1, max_size=5, max_lifetime=600, timeout=10,
            kwargs={"application_name": "literature-os-gradio"}
        )
    return _POOL

def run_sql(sql: str, params: list | tuple | None = None):
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.execute(sql, params or [])
            return cur.fetchall()

def load_year_bounds_and_tags():
    try:
        years = run_sql("""
            SELECT MIN(year), MAX(year)
            FROM papers
            WHERE year BETWEEN 1900 AND 2100
        """)
        min_year, max_year = years[0]
        min_year = int(min_year or 2010)
        max_year = int(max_year or 2025)
    except Exception:
        min_year, max_year = 2010, 2025

    try:
        rows = run_sql("SELECT DISTINCT tag FROM tags WHERE tag IS NOT NULL ORDER BY tag")
        tag_choices = [r[0] for r in rows]
    except Exception:
        tag_choices = []
    return min_year, max_year, tag_choices

def q_papers(tags, y_min, y_max, search, limit):
    sql = ["""
        SELECT
            p.id,
            COALESCE(p.title,'(No title)') AS title,
            COALESCE(p.journal,'(No journal)') AS journal,
            p.year,
            COALESCE(p.doi,'') AS doi,
            COALESCE(p.citation_count,0) AS citations
        FROM papers p
        WHERE 1=1
    """]
    params = []

    if tags:
        sql.append(" AND EXISTS (SELECT 1 FROM tags t WHERE t.paper_id = p.id AND t.tag = ANY(%s))")
        params.append(tags)

    if y_min is not None and y_max is not None:
        sql.append(" AND p.year BETWEEN %s AND %s")
        params.extend([int(y_min), int(y_max)])

    if search:
        q = f"%{search.strip()}%"
        sql.append(" AND (p.title ILIKE %s OR p.abstract ILIKE %s)")
        params.extend([q, q])

    sql.append(" ORDER BY p.year DESC NULLS LAST, p.citation_count DESC NULLS LAST LIMIT %s")
    params.append(int(limit))

    rows = run_sql("".join(sql), params)
    df = pd.DataFrame(rows, columns=["ID", "Title", "Journal", "Year", "DOI", "Citations"])
    if not df.empty:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
        # Keep DOI as plain URL text (no client-side link widget complexity)
        df["DOI"] = df["DOI"].map(lambda x: f"https://doi.org/{x.strip()}" if x and str(x).strip() else "")
    return df

def do_search(search, year_range, tags, limit):
    try:
        y_min, y_max = year_range
        df = q_papers(tags, y_min, y_max, search, limit)
        return df, gr.update(visible=True), gr.update(visible=True), ""
    except Exception as e:
        err = f"Query failed: {e}"
        return pd.DataFrame(), gr.update(visible=False), gr.update(visible=False), err

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## Literature OS")
    errbox = gr.Markdown(visible=False)

    # Load year bounds + tags at app start
    MIN_Y, MAX_Y, TAG_CHOICES = load_year_bounds_and_tags()

    with gr.Row():
        search = gr.Textbox(label="Search titles & abstracts", placeholder="ketamine depression…")
        year_range = gr.RangeSlider(MIN_Y, MAX_Y, value=[max(MIN_Y, MAX_Y-10), MAX_Y], step=1, label="Year range")
    tags = gr.CheckboxGroup(choices=TAG_CHOICES, label="Tags (optional)")
    with gr.Row():
        limit = gr.Slider(50, 1000, value=200, step=50, label="Row limit", interactive=True)
        run_btn = gr.Button("Search", variant="primary")

    out = gr.Dataframe(visible=False)
    dl_btn = gr.DownloadButton("Download CSV", visible=False, label="Download CSV")

    def prepare_download(df):
        # Gradio passes the whole DF; we write a temp CSV and give its path for download
        if df is None or getattr(df, "empty", True):
            return None
        path = "/tmp/papers.csv"
        df.to_csv(path, index=False)
        return path

    run_btn.click(
        fn=do_search,
        inputs=[search, year_range, tags, limit],
        outputs=[out, out, dl_btn, errbox],
        preprocess=False,
        postprocess=False,
    ).then(
        fn=prepare_download,
        inputs=[out],
        outputs=[dl_btn]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
