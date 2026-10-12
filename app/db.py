import sqlite3

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    name TEXT PRIMARY KEY,
    description TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    predicted_category TEXT NOT NULL,
    confidence REAL NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | corrected
    final_category TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(review_status);
"""

DEFAULT_CATEGORIES = {
    "invoice": "Bills and payment requests",
    "resume": "CVs and job applications",
    "contract": "Legal agreements",
    "research_paper": "Academic and scientific papers",
    "news_article": "News reports",
}


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    if con.execute("SELECT COUNT(*) FROM categories").fetchone()[0] == 0:
        con.executemany("INSERT INTO categories VALUES (?, ?)", DEFAULT_CATEGORIES.items())
    con.commit()
    con.close()
