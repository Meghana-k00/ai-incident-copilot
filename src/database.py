
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
        
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS approval_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                proposed_action TEXT NOT NULL,
                workflow_thread_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending',
                        'approved',
                        'rejected',
                        'changes_requested'
                    )),
                reviewer_id TEXT,
                decision_note TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                decided_at TEXT
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS approval_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                approval_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                actor_id TEXT,
                details TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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

def create_approval_request(
    incident_id: int,
    proposed_action: str,
    workflow_thread_id: str | None = None,
) -> dict | None:
    """Create a pending approval request for an existing incident."""
    action = proposed_action.strip()

    if not action:
        raise ValueError("Proposed action cannot be empty.")

    with get_connection() as connection:
        incident = connection.execute(
            "SELECT id FROM incidents WHERE id = ?",
            (incident_id,),
        ).fetchone()

        if incident is None:
            return None

        cursor = connection.execute(
            """
            INSERT INTO approval_requests (
                incident_id,
                proposed_action,
                workflow_thread_id
            )
            VALUES (?, ?, ?)
            """,
            (incident_id, action, workflow_thread_id),
        )

        approval_id = cursor.lastrowid

        connection.execute(
            """
            INSERT INTO approval_audit_log (
                approval_id, event_type, details
            )
            VALUES (?, ?, ?)
            """,
            (
                approval_id,
                "created",
                "Approval request created.",
            ),
        )

        row = connection.execute(
            "SELECT * FROM approval_requests WHERE id = ?",
            (approval_id,),
        ).fetchone()

        return dict(row)


def get_approval_request(approval_id: int) -> dict | None:
    """Retrieve an approval request by its ID."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM approval_requests WHERE id = ?",
            (approval_id,),
        ).fetchone()

        return dict(row) if row is not None else None


def decide_approval_request(
    approval_id: int,
    decision: str,
    reviewer_id: str,
    decision_note: str = "",
) -> dict | None:
    """Record one decision on a pending approval request."""
    allowed_decisions = {
        "approved",
        "rejected",
        "changes_requested",
    }

    if decision not in allowed_decisions:
        raise ValueError("Unsupported approval decision.")

    reviewer = reviewer_id.strip()
    if not reviewer:
        raise ValueError("Reviewer ID cannot be empty.")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE approval_requests
            SET status = ?,
                reviewer_id = ?,
                decision_note = ?,
                decided_at = CURRENT_TIMESTAMP
            WHERE id = ? AND status = 'pending'
            """,
            (
                decision,
                reviewer,
                decision_note,
                approval_id,
            ),
        )

        if cursor.rowcount == 0:
            return None

        connection.execute(
            """
            INSERT INTO approval_audit_log (
                approval_id,
                event_type,
                actor_id,
                details
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                approval_id,
                decision,
                reviewer,
                decision_note,
            ),
        )

        row = connection.execute(
            "SELECT * FROM approval_requests WHERE id = ?",
            (approval_id,),
        ).fetchone()

        return dict(row)


def get_approval_audit_log(
    approval_id: int,
) -> list[dict]:
    """Return audit events for an approval request in order."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM approval_audit_log
            WHERE approval_id = ?
            ORDER BY id
            """,
            (approval_id,),
        ).fetchall()

        return [dict(row) for row in rows]

def list_pending_approvals() -> list[dict]:
    """Return approval requests waiting for a decision."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM approval_requests
            WHERE status = 'pending'
            ORDER BY created_at, id
            """
        ).fetchall()

        return [dict(row) for row in rows]
