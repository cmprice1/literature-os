import os
from pyzotero import zotero
from db import upsert_paper, upsert_tags
from tag_rules import infer_tags
from enrichers import crossref_meta, unpaywall_oa, openalex_citations

ZOTERO_API_KEY = os.environ["ZOTERO_API_KEY"]
ZOTERO_LIBRARY_TYPE = os.environ["ZOTERO_LIBRARY_TYPE"]  # "user"|"group"
ZOTERO_LIBRARY_ID = os.environ["ZOTERO_LIBRARY_ID"]
INBOX_COLLECTION = os.environ.get("INBOX_COLLECTION", "Inbox")
UNPAYWALL_EMAIL = os.environ["UNPAYWALL_EMAIL"]

def get_client():
    return zotero.Zotero(ZOTERO_LIBRARY_ID, ZOTERO_LIBRARY_TYPE, ZOTERO_API_KEY)

def get_inbox_key(z):
    for c in z.collections():
        if c["data"]["name"] == INBOX_COLLECTION:
            return c["key"]
    return None

def main():
    print("🔗 Connecting to Zotero...")
    z = get_client()
    print("✅ Connected.")

    inbox_key = get_inbox_key(z)
    if not inbox_key:
        print(f"❌ Collection '{INBOX_COLLECTION}' not found")
        return
    print(f"📂 Found inbox collection: {inbox_key}")

    items = z.collection_items(inbox_key, limit=100)
    print(f"📄 Retrieved {len(items)} items from Zotero.")

    for i, it in enumerate(items, 1):
        d = it["data"]
        doi = d.get("DOI") or ""
        title = d.get("title") or ""
        print(f"\n--- Processing item {i}/{len(items)} ---")
        print(f"Title: {title}")
        print(f"DOI: {doi}")

        # augment metadata
        cr = crossref_meta(doi)
        print(f"Crossref: {cr}")
        oa = unpaywall_oa(doi, UNPAYWALL_EMAIL)
        print(f"Unpaywall: {oa}")
        cx = openalex_citations(doi)
        print(f"OpenAlex: {cx}")

        tags = set(infer_tags(d.get("publicationType") or [],
                              title or cr.get("title", ""),
                              d.get("abstractNote") or ""))
        tags.add("status:screened")
        print(f"Tags: {tags}")

        # write
        paper_id = upsert_paper(
            zotero_key=d["key"],
            doi=doi or None,
            pmid=None,
            title=title or cr.get("title") or "(untitled)",
            journal=cr.get("journal"),
            year=cr.get("year"),
            abstract=d.get("abstractNote"),
            oa_status=oa.get("oa_status"),
            best_pdf_url=oa.get("best_pdf_url"),
            citation_count=cx.get("citation_count"),
        )
        print(f"✅ Upserted paper ID {paper_id}")
        upsert_tags(paper_id, tags)
        print(f"✅ Tags upserted")


if __name__ == "__main__":
    main()
