"""
PostgreSQL database layer for CPLC Face Search.
Handles: person index metadata, search history.
Falls back gracefully if PostgreSQL is not available.
"""
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger("db")

_pool = None
_enabled = False

PG_DSN = os.environ.get(
    "CPLC_PG_DSN",
    "postgresql://cplc:cplc@localhost:5432/cplc_faces"
)


def _connect():
    """Create a raw psycopg2 connection (or None if unavailable)."""
    try:
        import psycopg2
        conn = psycopg2.connect(PG_DSN)
        conn.autocommit = True
        return conn
    except Exception as exc:
        logger.warning("PostgreSQL unavailable (%s) – running without DB.", exc)
        return None


def init_db():
    """Create tables if they do not exist. Call once at startup."""
    global _enabled
    conn = _connect()
    if conn is None:
        _enabled = False
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_index_meta (
                id          SERIAL PRIMARY KEY,
                db_root     TEXT        NOT NULL,
                model       TEXT        NOT NULL,
                detector    TEXT        NOT NULL,
                num_clusters INTEGER    NOT NULL DEFAULT 8,
                total_faces  INTEGER   NOT NULL DEFAULT 0,
                built_at    TIMESTAMP  NOT NULL DEFAULT NOW(),
                UNIQUE(db_root, model, detector, num_clusters)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_records (
                id          SERIAL PRIMARY KEY,
                index_id    INTEGER REFERENCES face_index_meta(id) ON DELETE CASCADE,
                rel_path    TEXT        NOT NULL,
                section     TEXT,
                added_at    TIMESTAMP  NOT NULL DEFAULT NOW(),
                UNIQUE(index_id, rel_path)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id          SERIAL PRIMARY KEY,
                query_name  TEXT,
                top_n       INTEGER,
                min_score   REAL,
                result_count INTEGER,
                results_json TEXT,
                searched_at TIMESTAMP  NOT NULL DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          SERIAL PRIMARY KEY,
                username    TEXT UNIQUE NOT NULL,
                password_hash TEXT      NOT NULL,
                role        TEXT        NOT NULL DEFAULT 'operator',
                created_at  TIMESTAMP  NOT NULL DEFAULT NOW()
            );
        """)
        conn.close()
        _enabled = True
        logger.info("PostgreSQL connected and tables initialised.")
        return True
    except Exception as exc:
        logger.warning("DB init failed: %s", exc)
        _enabled = False
        try:
            conn.close()
        except Exception:
            pass
        return False


def is_enabled():
    return _enabled


def record_index_build(db_root: str, model: str, detector: str,
                       num_clusters: int, total_faces: int,
                       relpaths: list, sections: dict) -> int | None:
    """Upsert an index build record. Returns index_id or None."""
    if not _enabled:
        return None
    conn = _connect()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO face_index_meta (db_root, model, detector, num_clusters, total_faces, built_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (db_root, model, detector, num_clusters)
            DO UPDATE SET total_faces=EXCLUDED.total_faces, built_at=NOW()
            RETURNING id;
        """, (db_root, model, detector, num_clusters, total_faces))
        index_id = cur.fetchone()[0]

        # Upsert per-file records
        if relpaths:
            rel_to_sec = {}
            for sec_name, members in sections.items():
                for (rel, _score) in members:
                    rel_to_sec[rel] = sec_name

            for rel in relpaths:
                cur.execute("""
                    INSERT INTO face_records (index_id, rel_path, section)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (index_id, rel_path)
                    DO UPDATE SET section=EXCLUDED.section;
                """, (index_id, rel, rel_to_sec.get(rel)))

        conn.close()
        return index_id
    except Exception as exc:
        logger.warning("record_index_build failed: %s", exc)
        try:
            conn.close()
        except Exception:
            pass
        return None


def record_search(query_name: str, top_n: int, min_score: float,
                  matches: list) -> None:
    """Log a search to the database for audit/history."""
    if not _enabled:
        return
    conn = _connect()
    if not conn:
        return
    try:
        cur = conn.cursor()
        # Keep only essential info to avoid huge JSON blobs
        slim = [{"rel": m["rel"], "sim": round(m["similarity"], 4),
                 "section": m.get("section", "")} for m in matches[:50]]
        cur.execute("""
            INSERT INTO search_history (query_name, top_n, min_score, result_count, results_json)
            VALUES (%s, %s, %s, %s, %s);
        """, (query_name, top_n, min_score, len(matches), json.dumps(slim)))
        conn.close()
    except Exception as exc:
        logger.warning("record_search failed: %s", exc)
        try:
            conn.close()
        except Exception:
            pass


def get_search_history(limit: int = 50) -> list:
    """Retrieve recent search history records."""
    if not _enabled:
        return []
    conn = _connect()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, query_name, top_n, min_score, result_count, searched_at
            FROM search_history
            ORDER BY searched_at DESC LIMIT %s;
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [
            {"id": r[0], "query_name": r[1], "top_n": r[2],
             "min_score": r[3], "result_count": r[4],
             "searched_at": r[5].isoformat() if r[5] else ""}
            for r in rows
        ]
    except Exception as exc:
        logger.warning("get_search_history failed: %s", exc)
        try:
            conn.close()
        except Exception:
            pass
        return []
