"""Small file-backed job store for background remediation status."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from .models import JobRecord, JobStatus, utc_now


class JobConflictError(RuntimeError):
    """A case already has a pending or running remediation."""

    def __init__(self, job: JobRecord) -> None:
        self.job = job
        super().__init__(f"case {job.case_id!r} already has active job {job.job_id}")


class JobStore:
    def __init__(
        self,
        root: Path,
        *,
        retention_days: int = 30,
        max_records: int = 1_000,
    ) -> None:
        self.root = root
        self.retention_days = retention_days
        self.max_records = max_records
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            for path in self.root.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    record = JobRecord.model_validate(payload)
                except Exception:
                    path.unlink(missing_ok=True)
                    continue
                if record.status in {JobStatus.PENDING, JobStatus.RUNNING}:
                    record.status = JobStatus.FAILED
                    record.error = "service restarted before the job completed"
                    record.updated_at = utc_now()
                    self._persist(record)
                self._jobs[record.job_id] = record
            self._prune_terminal()

    def _copy(self, record: JobRecord) -> JobRecord:
        return JobRecord.model_validate(record.model_dump(mode="json"))

    def _persist(self, record: JobRecord) -> None:
        target = self.root / f"{record.job_id}.json"
        temporary = self.root / f".{record.job_id}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            record.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def _remove(self, record: JobRecord) -> None:
        self._jobs.pop(record.job_id, None)
        (self.root / f"{record.job_id}.json").unlink(missing_ok=True)

    def _prune_terminal(self) -> None:
        cutoff = utc_now() - timedelta(days=self.retention_days)
        terminal = [
            record
            for record in self._jobs.values()
            if record.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        ]
        for record in terminal:
            if record.updated_at < cutoff:
                self._remove(record)

        retained = sorted(
            (
                record
                for record in self._jobs.values()
                if record.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
            ),
            key=lambda record: record.updated_at,
            reverse=True,
        )
        for record in retained[self.max_records :]:
            self._remove(record)

    def create(
        self,
        *,
        case_id: str,
        generation_model: str,
        validation_model: str,
    ) -> JobRecord:
        with self._lock:
            active = self.find_active_by_case(case_id)
            if active is not None:
                raise JobConflictError(active)
            record = JobRecord(
                job_id=uuid.uuid4().hex,
                case_id=case_id,
                status=JobStatus.PENDING,
                generation_model=generation_model,
                validation_model=validation_model,
            )
            self._jobs[record.job_id] = record
            self._persist(record)
            self._prune_terminal()
            return self._copy(record)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            self._prune_terminal()
            record = self._jobs.get(job_id)
            return self._copy(record) if record is not None else None

    def find_active_by_case(self, case_id: str) -> JobRecord | None:
        with self._lock:
            for record in self._jobs.values():
                if record.case_id == case_id and record.status in {
                    JobStatus.PENDING,
                    JobStatus.RUNNING,
                }:
                    return self._copy(record)
        return None

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            updated = record.model_copy(
                update={**changes, "updated_at": utc_now()},
            )
            updated = JobRecord.model_validate(updated.model_dump())
            self._jobs[job_id] = updated
            self._persist(updated)
            self._prune_terminal()
            return self._copy(updated)
