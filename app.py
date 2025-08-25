import os
import urllib.parse
import pandas as pd
import gradio as gr
from psycopg_pool import ConnectionPool

# ---------- DB Pool ----------
_POOL = None

def _ensure_sslmode(dsn: str) -> str:
    # Append sslmode=require if not present
    if "sslmode=" in dsn:
        return dsn
    sep = "&" if "?" in dsn else "?"
    return f"{dsn}{sep}sslmode=require"

def get_pool() -> ConnectionPool:
    global _POOL
    if _POOL is None:
        dsn = os.environ.get("DB_READER_DSN")
        if not dsn:
            raise RuntimeError("Missing DB_READER_DSN secret.")
        dsn = _ensure_sslmode(dsn)
        _POOL = ConnectionPool(
            conninfo=dsn,
            min_size=1, max_size=5, max_lifetime=600, timeout=10,
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

# ---------- Data helpers ----------
def load_year_bounds_and_tags():
    min_year, max_year, tag_choices = 2010, 2025, []
    try:
        years = run_sql("""
            SELECT MIN(year), MAX(year)
            FROM papers
            WHERE year BETWEEN 1900 AND 2100
        """)
        if years and years[0]:
            lo, hi = years[0]
            if lo: min_year = int(lo)
            if hi: max_year = int(hi)
    except Exception:
        pass

    try:
        rows = run_sql("SELECT DISTINCT tag FROM tags WHERE tag IS NOT NULL ORDER BY tag")
        tag_choices = [r[0] for r in rows]
    except Exception:
        pass

    # guardrails
    if min_year > max_year:
        min_year, max_year = 2010, 2025
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
        if y_min > y_max:
            y_min, y_max = y_max, y_min
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
        df["DOI"] = df["DOI"].map(lambda x: f"https://doi.org/{x.strip()}" if x and str(x).strip() else "")
    return df

def do_search(search, year_min, year_max, tags, limit):
    try:
        df = q_papers(tags, year_min, year_max, search, limit)
        return df, gr.update(visible=True), gr.update(visible=True), gr.update(value="", visible=False)
    except Exception as e:
        return (pd.DataFrame(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=f"**Query failed:** {e}", visible=True))

def prepare_download(df):
    if df is None or getattr(df, "empty", True):
        return None
    path = "/tmp/papers.csv"
    df.to_csv(path, index=False)
    return path

# ---------- UI ----------
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## Literature OS")
    errbox = gr.Markdown(visible=False)

    # Try to read bounds & tags; if DB auth is bad, we still render UI
    try:
        MIN_Y, MAX_Y, TAG_CHOICES = load_year_bounds_and_tags()
    except Exception as e:
        MIN_Y, MAX_Y, TAG_CHOICES = 2010, 2025, []
        errbox.value = f"Warning: could not load tags/years ({e}). Check DB_READER_DSN."

    with gr.Row():
        search = gr.Textbox(label="Search titles & abstracts", placeholder="ketamine depression…")
    with gr.Row():
        year_min = gr.Slider(MIN_Y, MAX_Y, value=max(MIN_Y, MAX_Y-10), step=1, label="Year min", interactive=True)
        year_max = gr.Slider(MIN_Y, MAX_Y, value=MAX_Y, step=1, label="Year max", interactive=True)
    tags = gr.CheckboxGroup(choices=TAG_CHOICES, label="Tags (optional)")
    with gr.Row():
        limit = gr.Slider(50, 1000, value=200, step=50, label="Row limit", interactive=True)
        run_btn = gr.Button("Search", variant="primary")

    out = gr.Dataframe(visible=False)
    dl_btn = gr.DownloadButton("Download CSV", visible=False, label="Download CSV")

    run_btn.click(
        fn=do_search,
        inputs=[search, year_min, year_max, tags, limit],
        outputs=[out, out, dl_btn, errbox],
        preprocess=False,
        postprocess=False,
    ).then(
        fn=prepare_download,
        inputs=[out],
        outputs=[dl_btn]
    )

    test_btn = gr.Button("Test DB connection")

    def db_ping():
        try:
            rows = run_sql("SELECT COUNT(*) FROM papers")
            n = rows[0][0] if rows else 0
            sample = run_sql("""SELECT id, title, year FROM papers ORDER BY year DESC NULLS LAST LIMIT 3""")
            df = pd.DataFrame(sample, columns=["ID", "Title", "Year"])
            msg = f"**papers count:** {n}"
            return gr.update(value=msg, visible=True), gr.update(value=df, visible=True)
        except Exception as e:
            return gr.update(value=f"**DB ping failed:** {e}", visible=True), gr.update(visible=False)

test_out_md = gr.Markdown(visible=False)
test_out_df = gr.Dataframe(visible=False)
test_btn.click(fn=db_ping, inputs=None, outputs=[test_out_md, test_out_df])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
