"""
Prompt registry.
"""

import sqlite3
import time
import os
from pathlib import Path
from dataclasses import dataclass


from utils.create_logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path("data/prompt_registry.db")



def _use_postgres():
    return os.getenv("USE_POSTGRES", "false").lower() == "true"

if _use_postgres:
    try: 
        import psycopg2
        import psycopg2.extras
    except Exception as e:
        logger.critical(f"PostgreSQL is not in env {e}")
        

def _get_conn():
    if _use_postgres():
        return psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            dbname=os.getenv("POSTGRES_DB"),
        )
    else:
        DB_PATH.parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        return conn


def _q(query: str) -> str:
    """Translate placeholders."""
    return query.replace("?", "%s") if _use_postgres() else query


def _execute(conn, query, params=()):
    if _use_postgres():
        with conn.cursor() as cur:
            cur.execute(_q(query), params)
            return cur
    else:
        return conn.execute(query, params)


def _fetchone(conn, query, params=()):
    if _use_postgres():
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_q(query), params)
            return cur.fetchone()
    else:
        return conn.execute(query, params).fetchone()



def init_db() -> None:
    with _get_conn() as conn:

        id_type = "SERIAL PRIMARY KEY" if _use_postgres() else "INTEGER PRIMARY KEY AUTOINCREMENT"

        _execute(conn, f"""
            CREATE TABLE IF NOT EXISTS evaluations (
                id            {id_type},
                trace_id      TEXT NOT NULL,
                task          TEXT NOT NULL,
                backend       TEXT NOT NULL,
                variant_a     TEXT NOT NULL,
                variant_b     TEXT NOT NULL,
                winner        TEXT NOT NULL,
                reachability_a REAL NOT NULL,
                reachability_b REAL NOT NULL,
                score_a       REAL NOT NULL,
                score_b       REAL NOT NULL,
                latency_a_ms  REAL NOT NULL,
                latency_b_ms  REAL NOT NULL,
                gap_report    TEXT,
                created_at    TEXT NOT NULL
            )
        """)

        _execute(conn, "CREATE INDEX IF NOT EXISTS idx_task ON evaluations(task)")
        _execute(conn, "CREATE INDEX IF NOT EXISTS idx_trace ON evaluations(trace_id)")

        _execute(conn, f"""
            CREATE TABLE IF NOT EXISTS optimization_trials (
                id              {id_type},
                run_id          TEXT NOT NULL,
                task            TEXT NOT NULL,
                backend         TEXT NOT NULL,
                base_prompt     TEXT NOT NULL,
                candidate_prompt TEXT NOT NULL,
                mutation        TEXT NOT NULL,
                trial_number    INTEGER NOT NULL,
                score           REAL NOT NULL,
                reachability    REAL NOT NULL,
                similarity      REAL NOT NULL,
                latency_ms      REAL NOT NULL,
                is_best         INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL
            )
        """)

        _execute(conn, "CREATE INDEX IF NOT EXISTS idx_optimization_run ON optimization_trials(run_id)")
        _execute(conn, "CREATE INDEX IF NOT EXISTS idx_optimization_task ON optimization_trials(task)")

        conn.commit()

    logger.info("Prompt registry initialized")



@dataclass
class EvalRecord:
    trace_id: str
    task: str
    backend: str
    variant_a: str
    variant_b: str
    winner: str
    reachability_a: float
    reachability_b: float
    score_a: float
    score_b: float
    latency_a_ms: float
    latency_b_ms: float
    gap_report: str = ""



def save(record: EvalRecord) -> int:
    '''Persist evaluation'''
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with _get_conn() as conn:
        cur = _execute(conn, """
            INSERT INTO evaluations (
                trace_id, task, backend,
                variant_a, variant_b, winner,
                reachability_a, reachability_b,
                score_a, score_b,
                latency_a_ms, latency_b_ms,
                gap_report, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.trace_id,
            record.task,
            record.backend,
            record.variant_a,
            record.variant_b,
            record.winner,
            record.reachability_a,
            record.reachability_b,
            record.score_a,
            record.score_b,
            record.latency_a_ms,
            record.latency_b_ms,
            record.gap_report,
            created_at,
        ))

        if _use_postgres():
            cur.execute("SELECT LASTVAL()")
            row_id = cur.fetchone()[0]
        else:
            row_id = cur.lastrowid

        conn.commit()

    return row_id



def best_variant_for_task(task: str) -> dict:
    '''Query'''
    with _get_conn() as conn:

        row = _fetchone(conn, """
            SELECT
                candidate_prompt AS best_template,
                AVG(score) AS avg_score,
                AVG(reachability) AS avg_reachability,
                COUNT(*) AS evaluations_sampled
            FROM optimization_trials
            WHERE task = ? AND is_best = 1
            GROUP BY candidate_prompt
            ORDER BY avg_score DESC
            LIMIT 1
        """, (task,))

        if not row:
            row = _fetchone(conn, """
                SELECT
                    CASE WHEN winner = 'a' THEN variant_a ELSE variant_b END AS best_template,
                    COALESCE(AVG(CASE WHEN winner = 'a' THEN score_a ELSE score_b END), 0.0) AS avg_score,
                    COALESCE(AVG(CASE WHEN winner = 'a' THEN reachability_a ELSE reachability_b END), 0.0) AS avg_reachability,
                    COUNT(*) AS evaluations_sampled
                FROM evaluations
                WHERE task = ?
                GROUP BY best_template
                ORDER BY avg_score DESC
                LIMIT 1
            """, (task,))

    if not row:
        return {}

    return {
        "task": task,
        "best_template": row["best_template"],
        "avg_reachability": round(float(row["avg_reachability"]), 4),
        "avg_score": round(float(row["avg_score"]), 4),
        "evaluations_sampled": row["evaluations_sampled"],
    }