
import sqlite3
from pathlib import Path
from contextlib import contextmanager


# Store the database in the project's data folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATA_DIR / "incidents.db"

@contextmanager
def get_connection():
    connection = sqlite3.connect(str(DATABASE_PATH))
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        
def init_db():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT NOT NULL,
                service_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open'
            )
            """
        )


def create_incident(title, description, severity, service_name):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO incidents
                (title, description, severity, service_name, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (title, description, severity, service_name, "open"),
        )
        incident_id = cursor.lastrowid
        row = connection.execute(
            "SELECT * FROM incidents WHERE id = ?",
            (incident_id,),
        ).fetchone()

        return dict(row)


def get_all_incidents():
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM incidents ORDER BY id"
        ).fetchall()

        return [dict(row) for row in rows]

def get_incident_by_id(incident_id):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM incidents WHERE id = ?",
            (incident_id,),
        ).fetchone()

        if row is None:
            return None

        return dict(row)


def delete_incident(incident_id):
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM incidents WHERE id = ?",
            (incident_id,),
        )

        return cursor.rowcount > 0


def update_incident_status(incident_id, status):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE incidents
            SET status = ?
            WHERE id = ?
            """,
            (status, incident_id),
        )

        if cursor.rowcount == 0:
            return None

        row = connection.execute(
            "SELECT * FROM incidents WHERE id = ?",
            (incident_id,),
        ).fetchone()

        return dict(row)
