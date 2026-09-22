"""Pesquisa segura de SoundFonts e samples."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

log = logging.getLogger(__name__)
CATALOG_PATH = Path("instrument_catalog.json")


def load_catalog() -> Dict[str, Any]:
    if CATALOG_PATH.is_file():
        try:
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.warning("catálogo corrompido: %s", e)
    return {"results": [], "sources": []}


def save_catalog(catalog: Dict[str, Any]) -> None:
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")


def register_source(*, name, url, license_name, commercial_use, format_="sf2", notes=""):
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL inválida")
    entry = {
        "name": name, "source": name, "url": url,
        "license": license_name, "commercial_use": bool(commercial_use),
        "format": format_, "notes": notes,
    }
    catalog = load_catalog()
    if not any(r.get("url") == url for r in catalog["results"]):
        catalog["results"].append(entry)
        save_catalog(catalog)
    return entry


def search_local(query):
    q = query.strip().lower()
    catalog = load_catalog()
    hits = []
    for r in catalog.get("results", []):
        blob = f"{r.get('name','')} {r.get('license','')} {r.get('notes','')}".lower()
        if q in blob:
            hits.append(r)
    return hits


def list_safe_sources():
    return [r for r in load_catalog().get("results", [])
            if r.get("license") and r.get("url")]
