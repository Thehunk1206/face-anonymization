"""SQLite holds upload jobs and the independent queue of published chunks."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

PIPELINE_VERSION = "2.2"
TERMINAL = ("completed", "needs_review", "failed")


class Jobs:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def initialize(self):
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                upload_id TEXT PRIMARY KEY, input_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued', frames_done INTEGER NOT NULL DEFAULT 0,
                total_frames INTEGER, error TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )""")
            existing = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            additions = {"pipeline_version": "TEXT", "settings_json": "TEXT", "filename": "TEXT",
                         "redaction_status": "TEXT", "redaction_stats": "TEXT",
                         "redaction_error": "TEXT", "checking_error": "TEXT"}
            for name, kind in additions.items():
                if name not in existing:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {kind}")
            # Finished historical jobs stay untouched; unfinished old jobs use the new pipeline.
            db.execute("""UPDATE jobs SET redaction_status='queued'
                WHERE pipeline_version IS NULL AND status IN ('queued','processing','checking')""")
            db.execute("""CREATE TABLE IF NOT EXISTS chunks (
                upload_id TEXT NOT NULL REFERENCES jobs(upload_id), chunk_index INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued', stats_json TEXT NOT NULL,
                frames_done INTEGER NOT NULL DEFAULT 0, flagged_frames INTEGER NOT NULL DEFAULT 0,
                suspected_faces INTEGER NOT NULL DEFAULT 0, error TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                PRIMARY KEY(upload_id, chunk_index)
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS chunks_queue ON chunks(status, created_at)")

    def get(self, upload_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE upload_id=?", (upload_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        job["chunks"] = self.chunks(upload_id)
        if job["pipeline_version"]:
            if job["status"] not in TERMINAL:
                job["status"] = {"queued": "queued", "processing": "processing",
                                 "completed": "checking", "failed": "failed"}[job["redaction_status"]]
            chunks = job["chunks"]
            job["checking_frames"] = sum(c["frames_done"] for c in chunks)
            job["checked_chunks"] = sum(c["status"] in ("passed", "flagged") for c in chunks)
            job["flagged_frames"] = sum(c["flagged_frames"] for c in chunks)
            job["checking_status"] = (
                "failed" if job["checking_error"] else
                "completed" if job["status"] in ("completed", "needs_review") else
                "checking" if any(c["status"] == "checking" for c in chunks) else
                "queued" if any(c["status"] == "queued" for c in chunks) else "waiting"
            )
        else:
            job.update(checking_frames=job["frames_done"], checked_chunks=0, flagged_frames=0,
                       checking_status="completed" if job["status"] in ("completed", "needs_review") else job["status"])
        return job

    def recent(self):
        with self.connection() as db:
            ids = db.execute("SELECT upload_id FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT 50").fetchall()
        return [self.get(row[0]) for row in ids]

    def add(self, upload_id, input_path, settings, filename=None):
        with self.connection() as db:
            result = db.execute("""INSERT OR IGNORE INTO jobs
                (upload_id,input_path,pipeline_version,settings_json,filename,redaction_status)
                VALUES (?,?,?,?,?,'queued')""",
                (upload_id, str(input_path.resolve()), PIPELINE_VERSION, json.dumps(settings), filename))
            return result.rowcount == 1

    def update(self, upload_id, **fields):
        allowed = {"status", "frames_done", "total_frames", "error", "redaction_status",
                   "redaction_stats", "redaction_error", "checking_error"}
        if not fields or not fields.keys() <= allowed:
            raise ValueError("Invalid job update")
        with self.connection() as db:
            db.execute(f"UPDATE jobs SET {','.join(f'{key}=?' for key in fields)}, "
                       "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE upload_id=?",
                       (*fields.values(), upload_id))

    def claim_upload(self, default_settings):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT * FROM jobs WHERE redaction_status='queued'
                ORDER BY created_at, rowid LIMIT 1""").fetchone()
            if row is None:
                return None
            db.execute("""UPDATE jobs SET redaction_status='processing',
                pipeline_version=COALESCE(pipeline_version,?),settings_json=COALESCE(settings_json,?)
                WHERE upload_id=?""", (PIPELINE_VERSION, json.dumps(default_settings), row["upload_id"]))
        return self.get(row["upload_id"])

    def recover(self, role):
        # Called only after taking that role's exclusive process lock.
        with self.connection() as db:
            if role == "redact":
                db.execute("UPDATE jobs SET redaction_status='queued' WHERE redaction_status='processing'")
            else:
                db.execute("UPDATE chunks SET status='queued',frames_done=0 WHERE status='checking'")

    def upgrade_checker(self, current, previous_models):
        """Keep redacted chunks; recheck unfinished older jobs with the current checker."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("""SELECT upload_id,settings_json,pipeline_version FROM jobs
                WHERE pipeline_version IN ('2.0','2.1')
                AND status NOT IN ('completed','needs_review','failed')""").fetchall()
            for row in rows:
                saved = json.loads(row["settings_json"])
                expected_models = previous_models if row["pipeline_version"] == "2.0" else current["models"]
                if saved["models"] != expected_models:
                    continue  # Unknown model combinations still fail the worker's version check.
                saved["models"] = current["models"]
                if row["pipeline_version"] == "2.0":
                    saved["settings"]["checker_size"] = current["settings"]["checker_size"]
                    saved["settings"]["checking_batch_size"] = min(
                        saved["settings"]["checking_batch_size"], current["settings"]["checking_batch_size"])
                db.execute("""UPDATE jobs SET pipeline_version=?,settings_json=?,status='queued',
                    checking_error=NULL,error=NULL,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    WHERE upload_id=?""", (PIPELINE_VERSION, json.dumps(saved), row["upload_id"]))
                db.execute("""UPDATE chunks SET status='queued',frames_done=0,flagged_frames=0,
                    suspected_faces=0,error=NULL WHERE upload_id=?""", (row["upload_id"],))

    def chunks(self, upload_id):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM chunks WHERE upload_id=? ORDER BY chunk_index", (upload_id,)).fetchall()
        result = []
        for row in rows:
            chunk = dict(row)
            chunk["stats"] = json.loads(chunk.pop("stats_json"))
            result.append(chunk)
        return result

    def publish_chunk(self, upload_id, index, stats):
        with self.connection() as db:
            db.execute("INSERT INTO chunks(upload_id,chunk_index,stats_json) VALUES(?,?,?)",
                       (upload_id, index, json.dumps(stats)))

    def claim_chunk(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT upload_id,chunk_index FROM chunks WHERE status='queued'
                ORDER BY created_at,upload_id,chunk_index LIMIT 1""").fetchone()
            if row is None:
                return None
            db.execute("UPDATE chunks SET status='checking',frames_done=0 WHERE upload_id=? AND chunk_index=?", tuple(row))
        return dict(row)

    def update_chunk(self, upload_id, index, **fields):
        if not fields or not fields.keys() <= {"status", "frames_done", "flagged_frames", "suspected_faces", "error"}:
            raise ValueError("Invalid chunk update")
        with self.connection() as db:
            db.execute(f"UPDATE chunks SET {','.join(f'{key}=?' for key in fields)} WHERE upload_id=? AND chunk_index=?",
                       (*fields.values(), upload_id, index))

    def ready_to_finalize(self):
        with self.connection() as db:
            row = db.execute("""SELECT upload_id FROM jobs j WHERE redaction_status='completed'
                AND status NOT IN ('completed','needs_review','failed')
                AND EXISTS (SELECT 1 FROM chunks c WHERE c.upload_id=j.upload_id)
                AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.upload_id=j.upload_id
                    AND c.status NOT IN ('passed','flagged')) ORDER BY created_at LIMIT 1""").fetchone()
        return self.get(row[0]) if row else None

    def queues(self):
        with self.connection() as db:
            uploads = dict(db.execute("SELECT redaction_status,count(*) FROM jobs WHERE redaction_status IS NOT NULL GROUP BY redaction_status"))
            chunks = dict(db.execute("SELECT status,count(*) FROM chunks GROUP BY status"))
        return {"anonymization": {"queued": uploads.get("queued", 0), "active": uploads.get("processing", 0)},
                "checking": {"queued": chunks.get("queued", 0), "active": chunks.get("checking", 0)}}
