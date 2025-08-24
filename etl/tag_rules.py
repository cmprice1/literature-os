import re

PUBTYPE_TO_DESIGN = {
    "Randomized Controlled Trial": "design:rct",
    "Meta-Analysis": "design:meta",
    "Systematic Review": "design:sr",
    "Case Reports": "design:case",
    "Clinical Trial": "design:trial",
}

KEYWORD_TO_DOMAIN = {
    r"\bpsychedelic|psilocybin|lsd|mdma|ayahuasca\b": "domain:psychedelics",
    r"\brtms|itbs|tdcs|ect\b": "domain:neuromod",
}

KEYWORD_TO_MODALITY = {
    r"\brtms\b": "modality:rTMS",
    r"\bitbs\b": "modality:iTBS",
    r"\btdcs\b": "modality:tDCS",
    r"\bect\b": "modality:ECT",
    r"\bketamine|esketamine\b": "modality:ketamine",
    r"\bpsilocybin\b": "modality:psilocybin",
    r"\bmdma\b": "modality:MDMA",
}

KEYWORD_TO_DX = {
    r"\btrd|treatment[- ]resistant depression\b": "dx:trd",
    r"\bmdd|major depressive disorder\b": "dx:mdd",
    r"\bptsd\b": "dx:ptsd",
    r"\bocd\b": "dx:ocd",
    r"\bsubstance use|addiction\b": "dx:sud",
}

def infer_tags(pub_types, title, abstract):
    tags = set()
    for t in pub_types or []:
        if t in PUBTYPE_TO_DESIGN:
            tags.add(PUBTYPE_TO_DESIGN[t])
    text = " ".join([title or "", abstract or ""]).lower()
    for pat, tag in {**KEYWORD_TO_DOMAIN, **KEYWORD_TO_MODALITY, **KEYWORD_TO_DX}.items():
        if re.search(pat, text):
            tags.add(tag)
    return sorted(tags)
