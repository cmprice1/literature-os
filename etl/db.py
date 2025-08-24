import os, psycopg
from psycopg.rows import dict_row

WRITER_DSN = os.environ["DB_WRITER_DSN"]

def get_conn():
    return psycopg.connect(WRITER_DSN, row_factory=dict_row)

def upsert_paper(**kw):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO papers (zotero_key, doi, pmid, title, journal, year, abstract, oa_status, best_pdf_url, citation_count)
            VALUES (%(zotero_key)s, %(doi)s, %(pmid)s, %(title)s, %(journal)s, %(year)s, %(abstract)s, %(oa_status)s, %(best_pdf_url)s, %(citation_count)s)
            ON CONFLICT (zotero_key) DO UPDATE SET
              doi=EXCLUDED.doi,
              pmid=EXCLUDED.pmid,
              title=EXCLUDED.title,
              journal=EXCLUDED.journal,
              year=EXCLUDED.year,
              abstract=EXCLUDED.abstract,
              oa_status=EXCLUDED.oa_status,
              best_pdf_url=EXCLUDED.best_pdf_url,
              citation_count=EXCLUDED.citation_count,
              updated_at=now()
            RETURNING id;
        """, kw)
        return cur.fetchone()["id"]

def upsert_tags(paper_id, tags):
    if not tags: return
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO tags (paper_id, tag) VALUES (%s, %s)
            ON CONFLICT (paper_id, tag) DO NOTHING;
        """, [(paper_id, t) for t in sorted(tags)])
