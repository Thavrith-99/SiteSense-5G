"""
Database connection helper for the PostGIS migration (Physical Finale).

Reads DATABASE_URL from, in order: the DATABASE_URL env var, then
Streamlit secrets (.streamlit/secrets.toml, gitignored). Nothing in
app.py imports this yet — coverage.py/recommend.py's file-based loaders
keep the deployed Streamlit Cloud app working untouched. This module is
the connection point data_prep/load_postgis.py and the future FastAPI
service layer build on.

Local dev:   export DATABASE_URL=postgresql://user:pass@localhost:5432/sitesense5g
Streamlit Cloud: add DATABASE_URL under Settings -> Secrets
DigitalOcean:    App Platform injects DATABASE_URL from the managed
                 Postgres add-on automatically.
"""
import os
import psycopg2


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    try:
        import streamlit as st
        return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    raise RuntimeError(
        "DATABASE_URL not set. Export it, or add it to "
        ".streamlit/secrets.toml locally / Streamlit Cloud Secrets in prod."
    )


def get_connection():
    """Return a new psycopg2 connection. Caller is responsible for closing it
    (or use as a context manager: `with get_connection() as conn:`)."""
    return psycopg2.connect(get_database_url())
