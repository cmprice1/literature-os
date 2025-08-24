import requests

def crossref_meta(doi):
    if not doi: return {}
    r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=20)
    if not r.ok: return {}
    m = r.json().get("message", {})
    return {
        "title": " ".join(m.get("title", []) or []),
        "journal": (m.get("container-title") or [""])[0],
        "year": (m.get("issued",{}).get("date-parts",[[None]])[0][0]),
    }

def unpaywall_oa(doi, email):
    if not doi: return {}
    r = requests.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": email}, timeout=20)
    if not r.ok: return {}
    j = r.json()
    best = j.get("best_oa_location") or {}
    return {"oa_status": j.get("oa_status"), "best_pdf_url": best.get("url_for_pdf") or best.get("url")}

def openalex_citations(doi):
    if not doi: return {}
    r = requests.get("https://api.openalex.org/works/doi:" + doi, timeout=20)
    if not r.ok: return {}
    j = r.json()
    return {"citation_count": j.get("cited_by_count")}
